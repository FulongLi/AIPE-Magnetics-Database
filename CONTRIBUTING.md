# Contributing magnetic records

Use a pull request with source rights, immutable original evidence, explicit SI
mapping, operating conditions and validation limitations. Do not fabricate real
material properties, loss curves or measurements. Preserve unknowns. Keep examples
marked synthetic and out of the real canonical catalogue.

Validate the complete related catalogue with `scripts/validate.py` so references
can resolve. Run `python -m unittest discover -s tests -v`. New domain fields need
dimensional and invalid-input tests plus a Core mapping/compatibility decision.
Changing schema semantics requires a version and migration note. Review changes;
do not auto-merge or overwrite previous evidence or published raw files.

Database queries/imports are separate from field-solver execution. Add future
solver support as tools/skills with explicit model scope, licences and validation
evidence. Prefer technically suitable open-source interfaces; do not vendor an
external connector without an explicit evaluated need.
