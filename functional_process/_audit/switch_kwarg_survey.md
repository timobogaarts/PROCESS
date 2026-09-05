# The switches that are still constructor kwargs — a survey, closed

**Closed, 2026-08-26 (conversion finished 2026-08-27).** Full conversion record and
outcome table: `next_steps_archive.md` §14.11. Of the 32 switch-carrying slots this
survey found, 30 are families now; only `power.component_thermal_powers` and
`power.delta_eta_step` still carry a static switch (kept static deliberately — see
below). Kept here: the measurement method, and the bugs/insights found along the way
that are not restated in §14.11.

## Method

A switch "invents an edge" when a node declares a read that is dead (provably unused,
via `jax.make_jaxpr` of the node's own callable seeded from a converged reference run)
at the switch's live value but live at some other value of the same switch's `IntEnum`.
For nodes gated by more than one switch, sweep the full product of their enums — a
single-switch sweep is blind to jointly-gated reads (`availability.electric_production`,
`availability.cplife_avail`, `costs.pf_magnet_cost` are only visible jointly). Where the
other arm raises rather than computing, its reads-set cannot be measured this way and is
read off the raise message / PROCESS source instead.

Bottom line at the time of the survey: of 345 declared reads across the 32 slots, **79
were dead-and-recoverable — invented edges — 23% of the surface**; a further 15 were dead
at every value of their own node's switches (a different, unrelated defect, since fixed
piecemeal per §14.11).

## Bugs found and fixed (not merely tidying — these were live incoherences)

- **`i_tf_sup` was resolved in seven different places** (a registry slot, five static
  kwargs, one port) that could disagree: `i_tf_sup = 0` assembled a machine that was
  resistive in `power.tf_power` but still superconducting in five other nodes. Fixed by
  threading one resolved value from `machine_from_indat` into all five kwarg sites.
- **`blktmodel`/`blkttype`/`ipowerflow`'s joint dispatch keys are arm *indices*, not
  switch *values***, and `machine_from_indat` was passing `blktmodel`'s raw value where
  an arm index was required, papered over with a default of `2` that is not even a legal
  `blktmodel` value. Silently selected the wrong ported arm for at least one real
  override (`blktmodel = 1` assembled the `ipowerflow == 0` shield-power node, then
  refused at the mass node for an unrelated reason). Fixed by deriving the arm index
  correctly in the factory.
- **`costs.cost_of_electricity`'s `ireactor`/`ipnet` is a node-*existence* condition, not
  a branch** — PROCESS never calls `coelc()` off that configuration, so the node "must
  not exist" per its own `__check_init__` docstring, yet the tree kept assembling it
  regardless. Fixed as an absent occupant (arm index 0 → `None`), not a refusal — cottax
  already spells absence, and an earlier, unrelated decision to remove all `X | None`
  slots had wrongly been read as blocking this.
- **`CoilsMass` answered `i_tf_sc_mat` with a module-level Python constant** baked into a
  `FromExactly(tfcoil.dcond[0])` default, not an `eqx.field(static=True)` — invisible to
  `switch_audit`, which only walks static fields, so an `i_tf_sc_mat = 5` machine could
  assemble a correct sibling node next to a coil-mass node still silently reading
  material 0. **The general lesson: a switch answered as a folded-in port default escapes
  every check built around static fields.** Fixed (eight occupants, keyed the same way as
  the sibling family); no general instrument for the class of bug was written, two cheap
  ones were sketched (a cross-slot "did any sibling's declared reads stay fixed while an
  occupant swapped" check; a source-level grep for constants named like a switch field).

## Design decisions worth keeping (why the obvious approach is wrong)

- **High-arity switches should not get one occupant per value.**
  `i_confinement_time` has 52 members, 49 reachable, and only **one** is live on any run
  this port models. The right shape is a family base holding the shared pre-dispatch
  logic, one occupant for the scaling actually exercised, and `UNPORTED` entries
  generated for the rest from the existing `elif` chain — not 49 classes.
- **A counterpart class existing does not mean it is the right shape to register.** Eight
  `Jcrit*` classes already implement `i_tf_sc_mat`'s arms, but the dispatch happens
  *inside* a 200-point sampling loop over an array (`winding_pack_curves`), where there
  is no scalar call site to bind a family occupant to. Conversion here means occupants of
  the *outer* node (`winding_pack_intersect_inputs`), not registering the existing
  per-material nodes directly.
- **"Convert only when reads differ" is not the criterion for every band** — reads-
  identical switches (`ife` × 7, and others where the only difference is that one arm
  raises) are still worth moving off the node body: the refusal belongs once, at
  assembly (`UNPORTED`), not seven times inside seven node bodies at trace time. Keep the
  reason strings; move where they're stated.
- **No switch on any of the 32 slots ever changed a node's declared *write* set** — a
  branch can zero an output or raise, never omit one. This is why none of the conversions
  cost a boundary-set change on the writes side; it only ever shrinks invented reads.

## Methodological point: the oracle for a hardcoded switch value is the converged
## `DataStructure`, not the input file

All 26 hardcoded switches were checked against PROCESS's own converged state for the
reference run, not just against what the IN.DAT sets — and one, `iohcl = 0`, is set by
neither the IN.DAT nor `machine_from_indat`: it comes from PROCESS's own stellarator
initialiser (`process/models/stellarator/initialization.py:24`). A test that only
compares against the input file can never see this class of value, and nothing would
notice if the initialiser changed it. Use the converged `DataStructure`, not the IN.DAT,
as the oracle for what a hardcoded switch value should be.

## What was left static, deliberately

`power.component_thermal_powers` and `power.delta_eta_step` are a genuine 2×3×2 product
of occupants over a 26-read signature (twelve classes / eight classes) that would only
remove two further dead reads on the reference machine. `next_steps_archive.md` §14.11
records this as the one case in the whole wave where the split's cost clearly exceeds
what it buys, and that the shape it actually wants is **nesting** (a sub-slot inside the
existing occupant), not a flat product — left for a pass that designs that.
