"""Magnetic Engineering State (MES) 0.3.0: model, JSON Schema, validation, round trip, migration and AIPE-Core adapter."""

from aipe_magnetics.state.core_adapter import mes_from_core_state, mes_to_core_state
from aipe_magnetics.state.migrate import MIGRATIONS, migrate_state
from aipe_magnetics.state.model import QV, STATE_SCHEMA, STATE_VERSION, MagneticEngineeringState
from aipe_magnetics.state.schema import (
    PACKAGED_SCHEMA,
    check_state_schema,
    dump_state,
    export_state_schema,
    load_state,
    state_json_schema,
    validate_state,
)

__all__ = ["MIGRATIONS", "MagneticEngineeringState", "PACKAGED_SCHEMA", "QV", "STATE_SCHEMA", "STATE_VERSION", "check_state_schema",
           "dump_state", "export_state_schema", "load_state", "mes_from_core_state", "mes_to_core_state", "migrate_state",
           "state_json_schema", "validate_state"]
