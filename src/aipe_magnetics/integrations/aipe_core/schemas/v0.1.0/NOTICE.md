# NOTICE: vendored AIPE-Core schemas (pinned copies)

Source repository: https://github.com/FulongLi/AIPE-Core
Upstream commit: 4e664f03df1e294fefc169244bf1223c7352e2be (committed 2026-10-08T16:45:17+01:00)
Upstream paths: `schemas/*.schema.json` and `examples/dab-10kw/engineering-state.json`
Licence: MIT (text reproduced verbatim below from the upstream `LICENSE` file at that commit)

These files are byte-exact copies of the committed upstream blobs (LF line endings, as
stored in Git) and are pinned for **offline compatibility testing only**. They are not a
release of AIPE-Core, and the logical `$id` URLs they carry
(`https://raw.githubusercontent.com/FulongLi/AIPE-Core/v0.1.0/schemas/...`) are identifiers,
not a promise that a release tag has been published. `aipe_magnetics.integrations.aipe_core`
resolves every `$id` to the local copy and never downloads a schema. To refresh the copies,
re-run `git show <commit>:<path>` from an AIPE-Core checkout and update the pinned SHA-256
digests in `state.py` (`VENDORED_SHA256`); the verification check
`integration.core-example-validates` fails when the copies drift from the pins.

## Reproduced source code (same licence)

In addition to the vendored files, `state.py` in the parent directory reproduces a substantial
portion of the upstream file `scripts/validate.py` at the same commit under this MIT licence:
`validate_engineering_state` is a near-verbatim port of `validate_state` (error wording kept so
that both validators can be compared line by line), together with `_walk`, `_offline_only`,
`state_validator` and the strict JSON loading rules (`load_json`: duplicate keys and non-finite
numbers rejected). `state.py` carries the copyright and licence notice in its header; this file
holds the full licence text. The `__main__.py` command line mirrors the upstream `main()` output
format (`PASS`/`FAIL` lines) but is an independent implementation.

| File | SHA-256 of vendored bytes |
| --- | --- |
| `common.schema.json` | `cf38dc02d8be1aacde0ef681c42baf5795d3f4b9e9d9650db5fb452841fbc622` |
| `control.schema.json` | `a14f8ecfa72d1b13d5631624574e1643a5a0053020237f2ee74146199200b05a` |
| `converter.schema.json` | `a4b11a66f7b36eaff9f334d1ff97e74c7e7f799c4d484ae364a29b14129a4996` |
| `engineering-state.schema.json` | `bdf2b50cb1d26aa2e77376c4c4335fd1ed150756465664756a7c8e07b691e39b` |
| `evidence.schema.json` | `1c2c17e11016823b0d4bb015ada4225ca1d204a2f955a8ad8c1425a5b2e6203d` |
| `magnetics.schema.json` | `42d1f1a6c41cf25f7f9c8b6671dc0486e71eaa45e8dba95afe3b985291919c79` |
| `operating-point.schema.json` | `bb437130a97ef671e199594bf06b79a792bcdff63b595ed19a97c5f914b105f4` |
| `passive.schema.json` | `19e48d3d1c0e13529269bb2eb59c1ec873b648da79902572de5ee6601e43e89d` |
| `project.schema.json` | `2ac5d03ae64a7cf5c9c07901c5eb60fc8ce98f5709e3a6de48ba0212fcd691b1` |
| `requirements.schema.json` | `33984772087763344ea6ee1d8d61e86136d5c3273a7dfdc0b3b2b63c2f2240be` |
| `semiconductor.schema.json` | `c28d021fcd060ee1f273741706b563726cda84ea02c6c2262a1ae2ee01ce304a` |
| `simulation.schema.json` | `9d9de270e8c61970beb8fe05dca88ab15f40ced3410514d1caae4b29b2072d55` |
| `thermal.schema.json` | `be012b04023c5026097bfe24e213f0937803b90bf0b1e489e0318a0fea24bc21` |
| `validation.schema.json` | `19784b542107b7ebe041c22524925997aa9483e87d9abae4357d7fb65155c5d6` |
| `examples/dab-10kw-engineering-state.json` | `0945d3a69727d72c4b47061b8cd52f6f1e422d63838ffe80c3a5a5480e4b15f0` |

## Upstream LICENSE (MIT), copied verbatim

```text
MIT License

Copyright (c) 2026 AIPE contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
