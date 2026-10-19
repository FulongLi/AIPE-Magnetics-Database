"""Combine the verification status of a design capability with that of the loss model it used.

A design result rests on two kinds of evidence: the closed-form sizing relations of the capability
itself (Faraday, reluctance model, Dowell, saturation screening ...) and the loss model that produced
the estimated loss values. The combined status is the *weaker* of the two so that a result never
looks better verified than its least verified ingredient.
"""

from __future__ import annotations

from typing import Any

from aipe_magnetics.common.results import VerificationStatus

_LEVEL = {"L0": 0, "L1": 1, "L2": 2, "L3": 3}
_STATE_RANK = {"verified": 0, "partially_verified": 1, "out_of_range": 2, "not_run": 3, "unverified": 4, "failed": 5}


def capability_status(registry: Any | None, capability: str, **conditions: Any) -> VerificationStatus:
    if registry is None:
        from aipe_magnetics.verification.suite import load_bundled_registry
        registry = load_bundled_registry()
    return registry.status_for(capability, **conditions)


def combine(design: VerificationStatus, model: VerificationStatus | None, *, design_label: str, model_label: str) -> VerificationStatus:
    """Weakest-link combination of two verification statuses with a statement naming both parts."""
    if model is None:
        return VerificationStatus(level=design.level, state=design.state, in_verified_range=design.in_verified_range,
                                  evidence_refs=list(design.evidence_refs),
                                  statement=f"{design_label}: {design.statement}")
    level = design.level if _LEVEL[design.level] <= _LEVEL[model.level] else model.level
    state = design.state if _STATE_RANK[design.state] >= _STATE_RANK[model.state] else model.state
    in_range: bool | None
    if design.in_verified_range is False or model.in_verified_range is False:
        in_range = False
    elif design.in_verified_range is None and model.in_verified_range is None:
        in_range = None
    else:
        in_range = True
    refs = list(dict.fromkeys([*design.evidence_refs, *model.evidence_refs]))
    return VerificationStatus(level=level, state=state, in_verified_range=in_range, evidence_refs=refs,
                              statement=f"{design_label}: {design.statement} {model_label}: {model.statement} "
                                        "The combined status is the weaker of the two.")
