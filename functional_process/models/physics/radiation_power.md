---
kind: model-unit
status: pending
---

## source
`process/models/physics/radiation_power.py` — `calculate_radiation_powers()`. Registry
unit #20 (see `../../_audit/unit_registry.md`) — scope limited to whatever
`stellarator.py`'s `st_phys()` (chunk 1B) actually calls; found via a bare
`import process.models.physics.radiation_power as physics_funcs` alias, not a
`self.<attr>.<method>` pattern, so missed by the original scoping grep.
