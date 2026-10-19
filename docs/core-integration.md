# AIPE-Core / AIPE-Registry integration

## Versions and compatibility

| Contract | Version | How it is satisfied |
| --- | --- | --- |
| AIPE-Core Engineering State | `0.1.0` (schemas pinned by commit in `src/aipe_magnetics/integrations/aipe_core/schemas/v0.1.0/NOTICE.md`) | executable export/import adapter + offline contract validator (`aipe_magnetics.integrations.aipe_core`) |
| AIPE Magnetic Data Schema | `0.2.0` (this repository, independent of the Core version) | `aipe_magnetics.data.records` / `schemas/v0.2` |
| AIPE-Registry capability manifest | `0.1.0` (`aipe.yaml`) | validated offline against the pinned Registry schema (`aipe_magnetics.integrations.aipe_registry`) and with the Registry's own validator in the ecosystem check |

`aipe.yaml` declares `compatibility.integration: native` because the round trip below is implemented
and tested; v0.1 was `mapped` (documentation only). The Registry schema is closed
(`additionalProperties: false`), so v0.2 metadata (model maturity, verification evidence, platforms,
interfaces, optional integrations, engineering limitations, standalone-operation guarantees) lives
under `extensions["aipe.magnetics"]`.

**Logical schema compatibility versus runtime installation:** Core and Registry are *schema and
contract* dependencies. Nothing at runtime needs an AIPE-Core checkout, an AIPE-Registry service, an
orchestrator or another domain expert. `aipe.yaml` therefore lists `aipe.core` as an *optional*
dependency; the pinned schema copies make the compatibility tests self-contained and offline.

## Round trip (tested)

```
EngineeringResult (design_transformer / design_inductor)
   │  aipe_magnetics.integrations.aipe_core.bridge.component_summary_from_design
   ▼
MagneticComponentSummary  (id, role, kind, design_status, turns_ratio / inductance / leakage, core_material_ref,
                           database_ref, evidence[assumption | analytical | source], limitations, detail)
   │  to_engineering_state(summary, created_at=..., project_id=..., requirements=...)
   ▼
Core Engineering State 0.1.0 (project, requirements with operating point, converter, magnetics[1], evidence[], validation)
   │  validate_engineering_state(state) == []        <- offline, same checks as Core's scripts/validate.py
   │  from_engineering_state(state) -> [MagneticComponentSummary]   (equal to the input summary)
   │  extract_requirements(state) -> design inputs
   ▼
```

Mapping rules follow Core's `docs/metadata.md` and `contracts/database-contract.md`:

| Magnetics concept | Core location |
| --- | --- |
| component kind (transformer, inductor, coupled_inductor, other) | `magnetics[].kind`, `role`, `design_status` |
| turns ratio, inductance, leakage inductance | `magnetics[]` SI quantities (`1`, `H`) |
| material record id, core record id | `core_material_ref`, `database_ref` (uri-references into this repository's records) |
| assumptions (flux limit, fill factor, currents ...) | evidence `kind: assumption` |
| the design calculation itself | evidence `kind: analytical` with `provenance.inputs` = the assumption / source evidence ids and `tool` attribution `aipe.magnetics@<version>` |
| datasheets and publications | evidence `kind: source` |
| flux density, losses, verification status, warnings | `extensions["aipe.magnetics-detail"]` (namespaced; Core fields are never overloaded) |
| converter requirements (power, voltages, frequency) | `requirements` + `operating_points[0]` with their own evidence |

The adapter never promotes an assumption to a measurement, never sets `validation.status` above
`analytically_checked`, and only emits `analytically_checked` when a passing analytical check with
analytical evidence is supplied. The demonstration writes the exported state to
`reports/demo/dab-10kw-engineering-state.json`.

## Ecosystem check with the real validators

`tests/integrations` contain optional tests (marker `external`) that run AIPE-Core's
`scripts/validate.py` and AIPE-Registry's `scripts/validate.py` as subprocesses when
`AIPE_ECOSYSTEM_CHECKOUTS=<dir containing AIPE-Core and AIPE-Registry>` is set. They are skipped with
an explicit reason otherwise. The release check for v0.2.0 was executed with both checkouts
(see `docs/release-verification-v0.2.0.md`).
