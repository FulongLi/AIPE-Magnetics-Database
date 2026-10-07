# Core v0.1 mapping and evidence provenance

AIPE-Core owns shared Engineering State; this database owns magnetics details.
The domain schema is independently versioned at `0.1.0`. Core compatibility is
`0.1.x`, integration `mapped`: this document specifies a mapping, not an executed
export path. Core's logical schema base is
`https://raw.githubusercontent.com/FulongLi/AIPE-Core/v0.1.0/schemas/`.
Until release publication, use a complete trusted local Core schema checkout.

| Magnetic domain | Engineering State location / rule |
| --- | --- |
| inductor / transformer / magnetic_component | `magnetics[]`, appropriate kind, explicit role and design_status |
| record ID + revisioned source URI | `id` / `database_ref`; retain independent domain version |
| inductance / leakage_inductance | Same named Core properties with `H` |
| turns_ratio | Core turns_ratio, dimensionless unit `1` |
| core_material | `core_material_ref` or a domain source artifact, without inventing properties |
| geometry / family / winding | Remain domain records referenced by a documented extension/artifact |
| operating_condition | Input context/evidence; Core operating point where dimensions match |
| source / assumption | Core evidence record of the corresponding kind with original reference |
| loss_model / measurement | Explicit artifacts and provenance; never promoted solely by schema validity |

A future executable adapter must add Core-required role, design status,
versioned tool attribution for simulation/measurement evidence and artifact
references. It must preserve source timestamps separately from conversion time,
retain unknown licence values and test against Core's offline validator.
Assumption evidence supporting a synthetic example must remain an assumption.
Neither material selection nor magnetic component suitability is automatic.

## Import lifecycle

1. Preserve original source bytes in `data/raw/` only when rights permit.
2. Record source URI/locator, owner, licence/access conditions and acquisition
   metadata. Use references when redistribution is not authorized.
3. Map fields and normalize units explicitly. The v0.1 JSON importer only accepts
   already normalized schema-valid records and adds a checksum of raw bytes.
4. Validate all related records together, including reference integrity and
   evidence. `--catalogue` supplies existing records during import.
5. Review the candidate before adding it to `data/canonical/`. Derived results
   live separately with methods, input IDs and versions. Never overwrite evidence.

SI spellings include `H`, `T`, `Hz`, `K`, `A`, `m`, `m^2`, `m^3` and `1`.
Turns are integral dimensionless counts. v0.1 supported physical properties are
positive; zero/bias sign conventions need a later explicit domain extension.
Missing quantities are omitted, not zero-filled. Loss coefficients are not
untyped floating values: this version references an external model artifact and
method rather than guessing coefficient dimensions or applicability ranges.

The validator rejects unknown record IDs but does not fetch cited artifacts,
authenticate documents or prove a measurement. The importer records source hash,
source filename and importer version; unchanged source inputs yield deterministic
candidate output. Existing provenance is never silently replaced.
