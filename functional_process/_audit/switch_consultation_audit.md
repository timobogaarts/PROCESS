# Does assembly actually consult the switches `UNPORTED` refuses?

**Measured 2026-08-31, all seven reference configurations.** `indat.UNPORTED`'s refusal
mechanism only fires where a `_slot_occupant` call for that field is reached — a node on
a branch that never asks the switch computes one arm unconditionally, and nothing catches
it being the wrong arm for the configuration. Swept all 50 dispatched switch axes (219
`UNPORTED` rows) four ways: traced which fields assembly actually consults per
configuration; force-set every refused value on every file and check whether it still
assembles (672 assemblies); diff each assembled graph's pinned static kwargs against the
file's own value; and, for axes that can't be swept this way (28 of 50 are derived arm
indices no `IN.DAT` names directly), grep PROCESS's real call path for whether it reads
the switch at all on that device.

**Result: the class is small — two real instances out of 50 axes, one fixed, one
latent** — and neither was a missing `UNPORTED` row. Both are hardcoded answers standing
where a refusal used to fire (before an unrelated change retired the precondition that
made the hardcode safe) or where no refusal was ever written because the other arm looked
like absence rather than a variant. **A table of `UNPORTED` refusals cannot catch either
class** — probe 3 (diff the assembled graph's static kwargs against the file) caught both
in seconds; this is the audit's actual finding, more than either bug.

## Fixed: `PfMagnetCost` was pinned to a central solenoid two files don't have

`indat.py` built the PF-magnet-cost occupant with `iohcl=PRESENT` hardcoded, a literal
left behind when an unrelated pass retired the `iohcl != 1` refusal that used to make the
hardcode safe. `spherical_tokamak_eval` and `st_regression` both set `iohcl=0` (no
central solenoid) — the PF coil *system* read that correctly, but the cost node didn't,
so both files costed 6 PF coils plus a nonexistent central solenoid where PROCESS costs 8
PF coils and no solenoid. Both configurations sat in the tracked matrix as "converged"
throughout — the error moves `.costs.*` and, on a cost-keyed objective, the objective
itself. Fixed by giving `PF_MAGNET_COST` a second occupant keyed on the same
`iohcl`/`supercond_cost_model` predicate the PF coil system already uses, so the two
can't disagree again. Confirmed against PROCESS's own cold-start cost term:
`spherical_tokamak_eval` moved 404.67 → 425.36 (PROCESS: 425.3643), `st_regression`
502.80 → 528.81 (PROCESS: 528.8146); a control tokamak with a real solenoid was
unchanged. Guarded going forward by
`test_switch_coverage.test_no_pinned_switch_contradicts_its_own_input_file`.

## Latent: two stellarator nodes assume a superconducting TF that nothing refuses

`i_tf_sup == 0` (resistive) is not in `UNPORTED` at all for a stellarator (only `== 2`,
aluminium, is refused) — but the only stellarator model ported for TF nuclear heating and
coil mass is explicitly the superconducting branch (`tf_nuclear_heating.py`'s own
docstring: "ports only the SUPERCONDUCTING branch"). Appending `i_tf_sup = 0` to
`stellarator_helias.IN.DAT` **assembles anyway** and silently keeps superconducting coil
mass/quench/stress nodes and superconducting TF nuclear heating on a machine declared
resistive. Not live today — no tracked file sets this — but invisible to any
`UNPORTED`-keyed check: the axis reports "enforced everywhere" because the one value
`UNPORTED` names *is* refused everywhere. A partially-enforced axis and a fully-enforced
one are indistinguishable in a table keyed on refusal rows; that's the structural lesson,
independent of this one instance.

## What was checked and found clean (so it isn't re-derived)

`i_blkt_coolant_type=2`/`i_fw_coolant_type='water'` assemble everywhere correctly because
the alternate arms are genuinely dead/unported code in PROCESS itself, not a port gap;
`ipowerflow`/`i_single_null`/`ife` all dispatch and refuse exactly where they should.  One
weaker latent noted, not fixed: `BUILDING_SIZING` pins `i_hcd_primary` rather than
threading the file's value, but no tracked file reaches that node, so it's dormant like
§ latent above, just less consequential.
