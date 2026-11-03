"""Inventory of every investigated source: integration status, pinned version, licence and the exact reason for
anything not integrated. ``data/sources.json`` and ``docs/source-catalogue.md`` are generated from this table.

Status values
    ``integrated``    normalised records are committed (``data/normalized/<directory>``)
    ``reference``     metadata/reference entry only; redistribution is not permitted or not established - a
                      user-supplied-file adapter or a pointer is provided instead
    ``unavailable``   no verified public dataset / dead link
"""

from __future__ import annotations

from typing import Any

SOURCES: list[dict[str, Any]] = [
    # ---- Priority A ---------------------------------------------------------------------------------------------
    {"id": "aipe_curated", "title": "AIPE curated datasheet catalogue (bundled with the package)", "priority": "A", "status": "integrated",
     "directory": None, "url": "https://github.com/FulongLi/AIPE-Magnetics-Database/tree/main/src/aipe_magnetics_database/data/catalog",
     "version": "0.2.0 catalogue", "license": "NOASSERTION", "redistribution": "values transcribed from manufacturer datasheets with locators",
     "notes": "11 materials (incl. 1 synthetic), 6 cores, 4 families, 4 manufacturers; Steinmetz sets fitted to MagNet are 'measurement_fit'."},
    {"id": "openmagnetics_mas", "title": "OpenMagnetics / PSMA MAS", "priority": "A", "status": "integrated", "directory": "mas",
     "url": "https://github.com/Power-Supply-Manufacturers-Association/MAS", "version": "c33ccfec8e00e156d632c9a52b02bc1cfdf714dc",
     "license": "Apache-2.0", "redistribution": "permitted (Apache-2.0); most values originate from manufacturer datasheets whose own terms "
                                                 "are not covered by the MAS licence",
     "importer": "aipe_magnetics_database.ingest.sources.mas",
     "notes": "core_materials, core_shapes, cores and the Git-LFS advanced_core_materials file; bobbins, wires and insulation materials "
              "are out of scope (winding components). MagNet-origin loss points are flagged duplicates of Princeton MagNet."},
    {"id": "princeton_magnet", "title": "Princeton MagNet (packaged scalar databases)", "priority": "A", "status": "integrated",
     "directory": "princeton-magnet", "url": "https://github.com/PrincetonUniversity/magnet", "version": "8a6892e91ac622e42f8b9182a6dfe9094ceea97a",
     "license": "MIT", "redistribution": "permitted (MIT; citation of the TPEL papers requested)",
     "importer": "aipe_magnetics_database.ingest.sources.princeton",
     "notes": "All 575,009 rows of the 10 packaged <material>_database.h5 files."},
    {"id": "princeton_magnet_waveforms", "title": "Princeton MagNet website B-H waveform downloads (<material>_cycle.zip / _measurements.zip)",
     "priority": "A", "status": "reference", "url": "https://mag-net.princeton.edu/", "version": "unpinned", "license": "MIT (website terms of access)",
     "redistribution": "permitted, but there is no stable URL or checksum (Streamlit per-session downloads)",
     "reason": "No pinnable download: the existing user-supplied adapter (PrincetonMagnetAdapter.import_website_zip) converts a manually "
               "downloaded zip into v0.2 waveform records with SHA-256; nothing is committed.",
     "adapter": "aipe_magnetics_database.data.sources.princeton_magnet.PrincetonMagnetAdapter"},
    {"id": "magnet_challenge_1", "title": "MagNet Challenge 2023 (MC1) data", "priority": "A", "status": "reference",
     "url": "https://github.com/minjiechen/magnetchallenge", "version": "repository 740fe6bb (MIT); data on Dropbox", "license": "NOASSERTION (data)",
     "redistribution": "unknown", "reason": "The 10-material training link is deleted; validation (256 MB) and testing (1.2 GB, materials A-E) "
                                            "Dropbox bundles carry no licence and are not pinned. The 10 training materials are the MagNet "
                                            "website data (see princeton_magnet_waveforms)."},
    {"id": "magnet_challenge_2", "title": "MagNet Challenge 2 / MagNetX database", "priority": "A", "status": "reference",
     "url": "https://github.com/minjiechen/magnetchallenge-2", "version": "ab4913ca63ebda1f8eab429c64d3e93f96204713", "license": "NOASSERTION",
     "redistribution": "not permitted without permission (no LICENSE file; contact pelsmagnet@gmail.com)",
     "reason": "No licence; the MagNetX database is a 24.8 GB unlicensed Dropbox archive. The existing read-only adapter "
               "(MagnetChallenge2Adapter) imports user-supplied files locally; nothing is downloaded or committed.",
     "adapter": "aipe_magnetics_database.data.sources.magnet_challenge2.MagnetChallenge2Adapter"},
    {"id": "paderborn_lea_materialdatabase", "title": "Paderborn University LEA materialdatabase", "priority": "A", "status": "integrated",
     "directory": "paderborn-lea", "url": "https://github.com/upb-lea/materialdatabase", "version": "04fff8e98ae3ddb72809223c42e8c939ed1073bb",
     "license": "GPL-3.0-only", "redistribution": "permitted under GPL-3.0 as a separate copyleft subtree (LICENSE, NOTICE, source offer)",
     "importer": "aipe_magnetics_database.ingest.sources.paderborn",
     "notes": "Independent LEA measurements, MagNet-derived permeability (derived_view), TDK MDT exports and digitised datasheet curves are "
              "classified separately; the 4 complex_permeability/datasheet files duplicate TDK_MDT and are not imported."},
    {"id": "paderborn_magnethub", "title": "Paderborn MagNet-Hub", "priority": "A", "status": "reference",
     "url": "https://github.com/upb-lea/mag-net-hub", "version": "1ed635d7bfac3c2aa024471af363beadc05f8932", "license": "MIT",
     "redistribution": "permitted", "reason": "Model zoo (trained MagNet Challenge models) plus two small test fixtures, not a measurement "
                                              "dataset; the fixture adapter stays available, model inference belongs to AIPE-Magnetics-Design.",
     "adapter": "aipe_magnetics_database.data.sources.paderborn_magnethub.MagnetHubAdapter"},
    {"id": "psma_core_loss_studies", "title": "PSMA / Dartmouth Core Loss Studies (Pilot, Phase II, Phase III)", "priority": "A",
     "status": "reference", "url": "https://www.psma.com/coreloss", "version": "unversioned web files (68 files, ~0.89 GB)",
     "license": "proprietary (PSMA copyright statement)",
     "redistribution": "not permitted: reproduction requires written PSMA consent (https://www.psma.com/legal/copyright-statement)",
     "reason": "Redistribution prohibited. A user-supplied-file adapter (aipe_magnetics_database.ingest.external.psma_coreloss_db) reads a "
               "locally downloaded Phase II coreloss.db into records that stay on the user's machine. Note: 'dPhi' is documented as peak "
               "flux [T] but its values are peak flux in Wb (peak B = dPhi / Ae); 'Ecyc' is energy per cycle of the whole core in J.",
     "adapter": "aipe_magnetics_database.ingest.external.psma_coreloss_db"},
    # ---- Priority B ---------------------------------------------------------------------------------------------
    {"id": "zenodo_8292861_hefmag", "title": "EMPIR 19ENG06 HEFMAG amorphous / nanocrystalline loss and complex permeability", "priority": "B",
     "status": "integrated", "directory": "hefmag", "url": "https://zenodo.org/records/8292861", "version": "doi:10.5281/zenodo.8292861 v1.0.0",
     "license": "CC-BY-4.0", "redistribution": "permitted with attribution", "importer": "aipe_magnetics_database.ingest.sources.zenodo",
     "notes": "Temperature and DC bias not stated; loss published as J/m^3 per cycle."},
    {"id": "zenodo_21626116_smc_ferrite_triangular", "title": "SMC and Mn-Zn ferrite loss under symmetric / asymmetric triangular polarization",
     "priority": "B", "status": "integrated", "directory": "zenodo-smc-ferrite-triangular", "url": "https://zenodo.org/records/21626116",
     "version": "doi:10.5281/zenodo.21626116", "license": "CC-BY-4.0", "redistribution": "permitted with attribution",
     "importer": "aipe_magnetics_database.ingest.sources.zenodo", "notes": "100 mT peak; temperature not stated."},
    {"id": "asu_mhz_core_loss_aggregate", "title": "ASU MAPEL MHz core-loss aggregate dataset", "priority": "B", "status": "reference",
     "url": "https://github.com/asu-mapel/MHzCoreLossAggregateDataset", "version": "8cf2378d72f0e3aa9cc96ee791ba267ee044faee",
     "license": "NOASSERTION (no LICENSE file)", "redistribution": "unknown - not redistributed (ask the maintainers)",
     "reason": "No licence. A user-supplied-file adapter (aipe_magnetics_database.ingest.external.asu_mhz_csv) normalises a local copy of "
               "MHzCoreLoss.csv (952 rows) and keeps the 13 original study labels as underlying_source so aggregated rows are never "
               "counted as independent of the original publications.",
     "adapter": "aipe_magnetics_database.ingest.external.asu_mhz_csv"},
    {"id": "amped_pitt_netl", "title": "AMPED / University of Pittsburgh / DOE-NETL high-frequency magnetics data", "priority": "B",
     "status": "unavailable", "url": "https://edx.netl.doe.gov/", "version": "-", "license": "-", "redistribution": "-",
     "reason": "No public tabular dataset with a verifiable licence was found (NETL EDX, OSTI, Zenodo searched); only two datasheet PDFs "
               "with unverified terms."},
    # ---- Priority C: manufacturers --------------------------------------------------------------------------------
    *[{"id": f"manufacturer_{key}", "title": f"{name} public engineering data", "priority": "C", "status": "reference", "url": url,
       "version": "-", "license": "proprietary", "redistribution": redistribution, "reason": reason,
       "adapter": "aipe_magnetics_database.ingest.external.user_table"}
      for key, name, url, redistribution, reason in [
          ("tdk", "TDK / EPCOS", "https://www.tdk-electronics.tdk.com/", "not permitted (terms of use s4.1)",
           "Magnetic Design Tool and datasheets may not be redistributed; 74 TDK materials reach the database through MAS."),
          ("ferroxcube", "Ferroxcube", "https://www.ferroxcube.com/", "not permitted (personal, non-commercial use only)",
           "Steinmetz coefficient spreadsheet and SFDT data are for personal use; 26 materials via MAS."),
          ("magnetics", "Magnetics Inc.", "https://www.mag-inc.com/", "unclear - treated as not redistributable",
           "Curve-fit tools are behind a contact form without licence; 88 materials via MAS."),
          ("micrometals", "Micrometals", "https://www.micrometals.com/", "unclear - treated as not redistributable",
           "Curve-fit coefficient workbook carries no licence; 93 materials via MAS."),
          ("fair_rite", "Fair-Rite", "https://fair-rite.com/", "unclear - treated as not redistributable",
           "Per-material complex-permeability CSV files carry no notice; 22 materials via MAS. Good candidate for a permission request."),
          ("vac", "VAC (Vacuumschmelze)", "https://vacuumschmelze.com/", "not permitted (imprint)", "PDF datasheets only; 2 materials via MAS."),
          ("metglas", "Metglas / Proterial", "https://metglas.com/", "not permitted (written permission required)", "PDF datasheets only."),
          ("tdg", "TDG", "https://www.tdgcore.com/", "unclear - treated as not redistributable", "PDF sheets only; 118 materials via MAS."),
          ("dmegc", "DMEGC", "https://www.dmegc.de/", "unclear - treated as not redistributable",
           "PDF sheets only (note: dmegc.com is not the manufacturer); 47 materials via MAS."),
          ("samwha", "Samwha", "https://www.samwha.com/", "unclear - treated as not redistributable", "PDF catalogues only; 4 materials via MAS."),
          ("coilcraft", "Coilcraft", "https://www.coilcraft.com/", "unclear (terms page not readable)", "S-parameter / SPICE models of components, not material data."),
          ("wurth", "Wurth Elektronik", "https://www.we-online.com/", "not permitted (non-sublicensable model licence; data mining banned)",
           "REDEXPERT / simulation libraries may not be redistributed; 12 materials via MAS."),
      ]],
]


def by_id(source_id: str) -> dict[str, Any]:
    return next(s for s in SOURCES if s["id"] == source_id)


def integrated() -> list[dict[str, Any]]:
    return [s for s in SOURCES if s["status"] == "integrated" and s.get("directory")]


__all__ = ["SOURCES", "by_id", "integrated"]
