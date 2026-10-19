# NOTICE: vendored AIPE-Registry capability manifest schema (pinned copy)

Source repository: https://github.com/FulongLi/AIPE-Registry
Upstream commit: f419260bc604e5e79ecde9d998139a39a5e0ebdf (committed 2026-10-08T16:45:43+01:00)
Upstream path: `schema/capability-manifest.schema.json`
Licence: MIT (text reproduced verbatim below from the upstream `LICENSE` file at that commit)

This file is a byte-exact copy of the committed upstream blob (LF line endings, as stored
in Git) and is pinned for **offline compatibility testing only**. It is not a release of
AIPE-Registry. `aipe_magnetics.integrations.aipe_registry.manifest` validates `aipe.yaml`
against this local copy and re-implements the essential loader and semantic rules of the
upstream `scripts/registry.py` (MIT) under attribution; it never fetches schemas or URLs. To
refresh the copy, re-run `git show <commit>:schema/capability-manifest.schema.json` from an
AIPE-Registry checkout and update `VENDORED_SHA256` in `manifest.py`.

| File | SHA-256 of vendored bytes |
| --- | --- |
| `capability-manifest.schema.json` | `6d0818650937010c12f5b33efcb1adf9246118f80d989ee7d1079fa8e6e24c5a` |

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
