# Naming convention: `data.<area>.<field>` → cottax `VarPath`

Draft. Revise as real units surface exceptions; note the exception here, don't just
work around it silently in one record.

## The base rule

`data.<area>.<field>` becomes a `VarPath` rooted at `<area>`, spelled as PROCESS already
spells it: `.physics.rmajor`, `.stellarator.powerht_constraint`. This costs nothing —
`process/data_structure/*_variables.py` module names are already the natural namespace,
and PROCESS's own naming convention (`documentation/source/development/standards.md`:
`<type>_<system>_<description>_<units>`) already carries almost everything cottax asks a
spelling to carry. Do not invent new names; port the existing one.

## Array elements

An `IterationVariable` with `array_index` (e.g. ID 125,
`f_nd_impurity_electron_array[2]`, display name `f_nd_impurity_electrons(03)`) becomes
`.impurity_radiation.f_nd_impurity_electron_array[2]` — a `SequenceKey`/index component
appended to the field's `VarPath`. **Record both** the storage path and the PROCESS
display label in the audit record; they must not be silently collapsed into one, since
existing PROCESS output (mfile, OUT.DAT labels) uses the display label.

## Switches are not ports

A switch (`i_*`, or a legacy name like `istell`) is a `data.<area>.<field>` value like
any other, but its **role** differs from an ordinary port:

- If it's read once at graph-*build* time to decide which node/subgraph to instantiate
  (a topology-changing switch, per `_audit/traceability_policy.md`), it is **not** a
  `VarPath` on any node at all — it's consumed by the Python code that assembles the
  `Graph`, the same way cottax expects `Graph` structure to be decided by the caller
  once, not re-read per evaluation.
- If it's kept as a static branch inside one node (a formula-changing switch with a
  provably identical reads-set across values — expected to be rare, see policy doc), it
  becomes a plain **static kwarg** on that node's `fn`, named after its PROCESS field
  name, not wrapped in a `VarPath`. `VarPath` is for values flowing along graph edges;
  a compile-time configuration choice is neither.

Every switch's audit record (`_audit/unit_registry.md` → per-switch rows) must say which
of these two it is, with the reads-set diff as justification — see
`traceability_policy.md`'s split-decision default.

## Function/module naming for the pure port

No convention fixed yet beyond: **the ported function's name should be derivable from
the `VarPath`(s) it owns** wherever there's a clear single output (e.g. a function
producing `.stellarator.powerht_constraint` might be named
`calculate_powerht_constraint`), following the `calculate_*` idiom PROCESS's own pure
cores already use (see `PlasmaDensityLimit.calculate_density_limit` in
`process/models/physics/density_limit.py`) — port the existing name where one already
exists rather than inventing a new one.

## Open questions (do not resolve silently in an individual audit record)

- Nested/sub-object reads like `self.physics.confinement.calculate_confinement_time(...)`
  — is `confinement` a namespace component (`.physics.confinement.*`) or does its output
  get a flat `VarPath` under `.physics.*` directly? Pending first real example from the
  pilot.
