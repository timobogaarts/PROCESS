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
~/miniconda3/envs/process_port/bin/python -m pytest functional_process
```

## Layout

- `_audit/` — the meta-documents: schema, naming convention, JAX-traceability policy,
  the unit registry (master list of what's in/out of scope and its status), and
  `test_harness.md` (the four-tier validation design, and what of it is built).
- `_harness/` — the validation machinery: tier contracts, PROCESS's finite-difference
  scheme and its error bar, sampling, tolerances. Design and rationale in
  `_audit/test_harness.md` § As built.
- `models/`, `core/solver/` — mirrors `process/`'s tree. Each audited unit gets one
  `<name>.md` record at the path corresponding to its source file (or, for constraints/
  switches, one record per registry entry inside `core/solver/`).
- `total_process.py` — every ported unit's `cottax` node, assembled by
  `graph_for(configuration)`. `GRAPH` is the graph PROCESS's own switch defaults produce;
  `render_xdsm.py` draws it to `xdsm.html` for inspection
  (`python -m functional_process.render_xdsm`). Both grow as units are ported; neither is
  a claim that the graph is complete.
- `configuration.py` — graph-assembly-time resolution of topology-changing switches, and
  the argument for why that is the only correct place for them. A node whose existence
  depends on a switch is declared as an `Alternative` under that switch rather than being
  registered unconditionally.

## Adding a unit

One unit is one stem, three files in the same directory: `<name>.md` (audit record,
first), `<name>.py` (the port), `test_<name>.py` (the case). The case declares the
PROCESS reference, the port, and the sample points, then subclasses the contract for the
tier its record assigns — it does not write test functions. Copy
`models/stellarator/test_density_limits.py`; it is the worked example.

If the unit's node only exists for some values of a switch, register it as an
`Alternative` in `total_process.TOPOLOGY_SWITCHES` rather than in `COMMON` — two nodes
that own the same output cannot both be in one graph, and `to_graph` will say so. Adding
the arm is what makes it reachable; `test_configuration.py` fails on an arm no
configuration selects.

## Scope (current)

Stellarator only (`data.stellarator.istell != 0` pipeline), not tokamak, not IFE.
Concretely: all of `process/models/stellarator/**`, plus only the specific methods
`Stellarator.run()` calls into on its injected sub-models (physics, power, hcpb,
buildings, vacuum, availability, costs, plasma_profile, neoclassics) — not those files'
entirety. See `_audit/unit_registry.md` for the exact list and the reasoning.
