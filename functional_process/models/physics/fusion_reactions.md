---
kind: model-unit
status: pending
---

## source
`process/models/physics/fusion_reactions.py` — `FusionReactionRate`
(`.deuterium_branching()`, `.calculate_fusion_rates()`, `.set_physics_variables()`),
`beam_fusion()`, `set_fusion_powers()`. Registry unit #19 (see
`../../_audit/unit_registry.md`) — scope limited to whatever
`stellarator.py`'s `st_phys()` (chunk 1B) actually calls; found via a bare
`import process.models.physics.fusion_reactions as reactions` alias, not a
`self.<attr>.<method>` pattern, so missed by the original scoping grep.
