"""Physical-property library (thermal and bulk properties the v0.2 record standard has no field for).

The values live in ``aipe_magnetics/data/reference/physical-properties.json`` with a basis, a source and,
where the literature spread is large, explicit lower/upper bounds. :func:`physical_property` returns a
:class:`~aipe_magnetics.common.quantity.Quantity` whose ``uncertainty`` carries those bounds (``kind='bounds'``)
or ``None`` when the source gives a single value. Nothing here is invented at run time: an unknown material or
property raises :class:`~aipe_magnetics.common.errors.DataNotFoundError`.
"""

from __future__ import annotations

import functools
import json
from pathlib import Path
from typing import Any

from aipe_magnetics.common.errors import DataNotFoundError, InvalidInputError
from aipe_magnetics.common.quantity import Quantity, Uncertainty
from aipe_magnetics.common.results import SourceReference

PROPERTIES_FILE = Path(__file__).resolve().parents[1] / "data" / "reference" / "physical-properties.json"

_SOURCE_KINDS = {"datasheet", "publication", "assumption", "repository", "dataset", "standard"}


@functools.lru_cache(maxsize=1)
def _library() -> dict[str, Any]:
    with PROPERTIES_FILE.open(encoding="utf-8") as handle:
        return json.load(handle)


def property_materials() -> list[str]:
    """Identifiers of the property-library entries (e.g. ``ferrite_mnzn``, ``copper``, ``winding_pack_litz``)."""
    return sorted(_library()["materials"])


def property_entry(material: str) -> dict[str, Any]:
    lib = _library()
    key = lib.get("material_class_map", {}).get(material, material)
    try:
        return lib["materials"][key]
    except KeyError:
        raise DataNotFoundError(f"no physical-property entry {material!r}; available: {', '.join(property_materials())}",
                                details={"material": material}) from None


def physical_property(material: str, name: str) -> Quantity:
    """``Quantity`` of property ``name`` of library entry (or catalogue material class) ``material``."""
    if not isinstance(material, str) or not isinstance(name, str):
        raise InvalidInputError("material and property name must be strings")
    entry = property_entry(material)
    try:
        item = entry[name]
    except KeyError:
        raise DataNotFoundError(f"property {name!r} is not recorded for {material!r}",
                                details={"material": material, "property": name}) from None
    uncertainty = None
    if "lower" in item and "upper" in item:
        uncertainty = Uncertainty(kind="bounds", lower=float(item["lower"]), upper=float(item["upper"]),
                                  basis=f"{item.get('basis', 'unspecified')} ({item.get('source', 'no source')})")
    return Quantity(value=float(item["value"]), unit=item["unit"], uncertainty=uncertainty)


def property_source(material: str, name: str) -> SourceReference:
    """Source reference of one library value (for result envelopes)."""
    entry = property_entry(material)
    item = entry.get(name)
    if item is None:
        raise DataNotFoundError(f"property {name!r} is not recorded for {material!r}")
    source_id = item.get("source")
    raw = _library()["sources"].get(source_id, {})
    kind = raw.get("kind", "assumption")
    return SourceReference(id=f"ref.{source_id}" if source_id else f"ref.{material}.{name}",
                           kind=kind if kind in _SOURCE_KINDS else "assumption",
                           reference=raw.get("reference", "unspecified"), url=raw.get("url"),
                           note=item.get("note") or raw.get("note"), license="NOASSERTION")


def describe_entry(material: str) -> dict[str, Any]:
    """JSON-serialisable description of one entry (values, units, bounds, basis, sources)."""
    entry = property_entry(material)
    out: dict[str, Any] = {"id": material, "name": entry.get("name"), "role": entry.get("role"), "properties": {}}
    for key, item in entry.items():
        if isinstance(item, dict) and "value" in item:
            out["properties"][key] = {k: item[k] for k in ("value", "unit", "lower", "upper", "basis", "note") if k in item}
            out["properties"][key]["source"] = property_source(material, key).model_dump(mode="json", exclude_none=True)
    return out


__all__ = ["PROPERTIES_FILE", "describe_entry", "physical_property", "property_entry", "property_materials", "property_source"]
