# functional_process

Audit trail and (eventually) source for a pure-functional, cottax-shaped rewrite of
PROCESS's **stellarator pipeline**. Mirrors `process/`'s directory layout at the paths
that are actually in scope.

- **`process/` is never modified.** Everything here is either analysis of `process/`
  (Phase 0) or new pure-functional/JAX code written from scratch, informed by that
  analysis (later phases).
- Read `../CLAUDE.md` first — it explains why this exists, the `process_port` env, and
  the cottax vocabulary (`VarPath`, `Compare`, `Cut`, `Drive`, tiers 1-4) used
  throughout `_audit/`.
- **Current phase: Phase 0 (pre-coding audit), stellarator.py (unit #1) audited in full —
  plus the validation harness and its first ports.** Standing practice: a chunk found to
  be tier-1 and self-contained is ported (function + `## cottax node` wrap + harness
  test) as soon as its audit lands, not batched for later. See
  `_audit/unit_registry.md` for the per-unit list and rationale, and
  `_audit/next_steps.md` for what is next — neither is summarised here, so that this
  file cannot drift away from them.

```bash
~/miniconda3/envs/process_port/bin/python -m pytest tests/functional_process
```

## Layout

- `_audit/` — the meta-documents, flat: schema, naming convention, JAX-traceability
  policy, the unit registry (master list of what's in/out of scope and its status), and
  `test_harness.md` (the four-tier validation design, and what of it is built).
- `_audit/units/` — the per-unit audit records, mirroring this tree record for record:
  one `<name>.md` at the path corresponding to its source file (or, for constraints/
  switches, one record per registry entry under `core/solver/`). They are bound to their
  units by `_audit/unit_registry.md`, which names each record's path, not by sitting next
  to the module — see `_audit/test_harness.md` § As built for why that changed.
- `_harness/` — the validation machinery: tier contracts, PROCESS's finite-difference
  scheme and its error bar, sampling, tolerances. Design and rationale in
  `_audit/test_harness.md` § As built.
- `models/`, `core/solver/` — the ports themselves, mirroring `process/`'s tree.
- `../tests/functional_process/` — the harness cases, mirroring this tree file for file,
  with `conftest.py` (markers, `--fp-fuzz`, sample parametrization) at its root.
- `models/<subsystem>/namespace.py` — the subsystem's naming scope: the `ModelNamespace`
  classes whose slots name that subsystem's ported nodes. They sit beside the models they
  name, and none of them reads a switch.
- `total_process.py` — `StellaratorProcess`, the whole device: one slot per subsystem,
  and nothing else. Not one `i_*` integer anywhere in it.
- `indat.py` — PROCESS's input encoding, and the one place this port reads it:
  `switches_from_indat`, the registries mapping each switch value to the occupant it
  selects, `UNPORTED`'s recorded refusals, and `machine_from_indat`, which assembles the
  `StellaratorProcess` an IN.DAT describes. `graph_for(machine)` and `GRAPH` (the
  reference IN.DAT's graph) live here for the same reason; `render_xdsm.py` draws `GRAPH`
  to `xdsm.html` for inspection (`python -m functional_process.render_xdsm`). Both grow
  as units are ported; neither is a claim that the graph is complete.

## Adding a unit

One unit is one stem, three files at the same relative path in three trees:
`<name>.md` (audit record, first) under `_audit/units/`, `<name>.py` (the port) here, and
`test_<name>.py` (the case) under `../tests/functional_process/`. So the record for
`models/stellarator/density_limits.py` is
`_audit/units/models/stellarator/density_limits.md` and its case is
`../tests/functional_process/models/stellarator/test_density_limits.py`. Add the record's
path to `_audit/unit_registry.md` as you write it — that row, not the file's location, is
what binds record to unit, and `test_registry_coverage.py` fails on a record the registry
does not name and on a registry row whose record is missing.

The case declares the PROCESS reference, the port, and the sample points, then subclasses
the contract for the tier its record assigns — it does not write test functions. Its
`audit_record` is the record's path relative to `_audit/units/`, i.e. the mirrored path,
the same string in every tree. Copy
`../tests/functional_process/models/stellarator/test_density_limits.py`; it is the worked
example.

A node is registered by naming it in a slot of its subsystem's
`models/<subsystem>/namespace.py`. If it only exists for some values of a switch, the
slot is annotated with the union of its occupants and left without a default, and the
registry mapping that switch's values to occupants goes in `indat.py` — two nodes that
own the same output cannot both be in one graph, and `to_graph` will say so. The tree
knows no switches and `indat.py` knows them all; `test_switch_coverage.py` and
`test_machine.py` are what check that split holds.

## Scope (current)

Stellarator only (`data.stellarator.istell != 0` pipeline), not tokamak, not IFE.
Concretely: all of `process/models/stellarator/**`, plus only the specific methods
`Stellarator.run()` calls into on its injected sub-models (physics, power, hcpb,
buildings, vacuum, availability, costs, plasma_profile, neoclassics) — not those files'
entirety. See `_audit/unit_registry.md` for the exact list and the reasoning.
