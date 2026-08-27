"""Block-by-block comparison: does `functional_process.mda`'s driven graph reproduce
a real, converged PROCESS run's own values?

Two halves, deliberately separable (`converged_data` is specific to one input file;
`compare` is not):

- `converged_data(input_file)` runs PROCESS's own `SingleRun` in-process and returns
  the live, converged `DataStructure` -- not a round trip through `MFILE.DAT` (loses
  precision) or the CLI (re-parses text).
- `compare(graph, data)` seeds `functional_process.mda`'s `Schedule` for `graph` from
  `data`'s own values (every boundary input, and every driven block's starting guess
  -- same converged run, per this session's established principle: we are checking
  whether the graph reproduces an answer PROCESS already found, not solving cold),
  runs it, and diffs every value the schedule produced against `data`'s own value at
  the same field.

`compare` also runs `switch_audit(graph, data)` -- a check on the graph's *registration*
rather than its arithmetic. Every `eqx.field(static=True)` switch kwarg a node carries
is read back off the assembled graph and compared with the same-named field on the
converged `DataStructure`. This exists because four separate bugs of exactly one shape
have been found in this project so far (`i_confinement_time`,
`i_thermal_electric_conversion`, `i_p_coolant_pumping`, `i_plasma_ignited` --
`_audit/next_steps.md` §8), each a hardcoded value copied from a
`process/data_structure/*_variables.py` bare Python default instead of from the run
being modelled, and each found only by luck when a downstream value happened to diverge
loudly. Nothing checked them directly before this.

**The audit is enum-aware.** Every model-selection static kwarg is typed with an
`IntEnum` (`_audit/model_tree_design.md` §4; the upstream one where PROCESS declares
one, `functional_process/models/switch_enums.py`'s otherwise), so a mismatch is reported
in the vocabulary the mistake was actually made in -- `PROCESS_1990 != KOVARI_2014`
rather than `0 != 1`. The comparison itself is unchanged and still numeric: `IntEnum`
members equal their `int` values, so the check is exactly as strict as before and works
identically against a `DataStructure` field that stores a bare `int`. The two static
kwargs that are deliberately *not* switches -- shape/resolution counts and set
membership, `_audit/switch_elimination_design.md` §3(b)/(c) -- are classified as such in
the report rather than passing silently as anonymous integers.

**`DuctDiameterRootFind` is excluded.** Confirmed directly in its own docstring
(`vacuum.py:334-344`): every one of its `VarPath`s (`l1`, `l2`, `l3`, `xmult_i`,
`ceff_i`, `d_duct`) is minted, not a real PROCESS `data` field -- in real PROCESS
these are locals inside `_solve_vacuum_pumping_old`'s per-species Newton loop, never
written to `DataStructure`. There is nothing in a converged run to seed or compare
this node against; it is a deliberate island (`VacuumOld`, not this node, is the
PROCESS-faithful, registered vacuum path). `compare` drops it (and its own
`^problem[...]` partner) before running.
"""

import dataclasses
import os
import pickle
import sys
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from cottax.blocking import Blocking
from cottax.evaluate import schedule_for
from cottax.plan import Delete
from cottax.spec import NodePath, VarPath

from functional_process.mda import driven_graph, starts_for

EXCLUDED_NODE_NAMES = ("duct_diameter_root_find",)
"""`.vacuum.duct_diameter_root_find`: see this module's own docstring -- no real
`DataStructure` field backs any of its `VarPath`s.

Matched with `in` against a node's `path_str()`, so the entry is the **slot name**, not
the occupant class: since `model_tree_design.md` §8 step 3 a node is named by where it
sits in the machine tree, and the class name no longer appears in the name at all.

**The coil island (`Intersect`, the `winding_pack_intersect_inputs` occupant and
`WindingPackTotalSizePost`) used to be here too, and is not any more** -- see
`_audit/closed/constraint_32_investigation.md` for the full evidence. The exclusion rested on
one true fact (`.stellarator.wp_width_r_min`, `Intersect`'s own unknown, is minted, so
`_ground_truth` fell back to the `0.0` placeholder, which is a bad enough starting
guess that `NewtonDriver`'s `optx.root_find` failed within `max_steps` and aborted the
whole `schedule()` call) and one false one ("nothing here to compare against" --
`WindingPackTotalSizePost` owns *eleven* real `DataStructure` fields
(`coils/calculate.py:1122-1136`) and only two mints). The true half is fixed at its
cause by the three `KNOWN_MINT_VALUES` entries below: `wp_width_r_min` is seeded from
`.tfcoil.dr_tf_wp_with_insulation`, which PROCESS *does* store and which is that same
crossing point after the (here inactive) turn-size clamp. With that seed the root find
converges and the whole island runs.
"""

KNOWN_MINT_VALUES = {
    # `.stellarator.coilcurrent` -- a local in `st_coil` (`process/models/stellarator/
    # coils/calculate.py:46,378`), never stored, but exactly recoverable from two real
    # fields: `calculate.py:276` writes `data.tfcoil.c_tf_total = data.tfcoil.n_tf_coils
    # * coilcurrent * 1.0e6`, so this inverts that same line. This port already
    # implements the identity in the same direction at `coils/quench.py:201`.
    ".stellarator.coilcurrent": (
        lambda d: d.tfcoil.c_tf_total / (d.tfcoil.n_tf_coils * 1.0e6)
    ),
    # `.impurity_radiation.pden_impurity_rad_total_mw` -- an `ImpurityRadiation`
    # instance attribute (`process/models/physics/impurity_radiation.py:667-668,737`),
    # never stored, but inverted from `process/models/physics/radiation_power.py:132`
    # (`pden_plasma_rad_mw = imp_rad.pden_impurity_rad_total_mw + pden_plasma_sync_mw`).
    #
    # **The `pden_impurity_core_rad_total_mw` sibling deliberately gets no entry.** Its
    # analogous inverse (`radiation_power.py:128-129`) is *not* valid: PROCESS clips
    # `.physics.pden_plasma_core_rad_mw` at zero after storing it
    # (`process/models/stellarator/stellarator.py:2153-2155`), so the stored value is
    # not recoverable back through the sum. `pden_plasma_rad_mw` itself is written
    # unclipped (`ibid.:2151`), which is what makes the entry above sound.
    ".impurity_radiation.pden_impurity_rad_total_mw": (
        lambda d: d.physics.pden_plasma_rad_mw - d.physics.pden_plasma_sync_mw
    ),
    # `.physics.pden_plasma_core_rad_mw_unclipped` / `..._outer_...` --
    # `PlasmaRadiationPowers`'s outputs *before* `st_phys`'s zero-clip
    # (`stellarator.py:2152-2159`), which `ClippedRadiationPowers` applies. PROCESS
    # overwrites the field in place, so only the post-clip value is ever stored: these
    # two mints equal the stored field **exactly whenever the clip is inactive**, and
    # are a lower bound on it otherwise -- the same discipline, and the same caveat, as
    # `.stellarator.wp_width_r_min` below. Measured on this run: core `0.0575`, outer
    # `0.0553`, both positive, so both are exact here. A run that clipped would show up
    # as a disagreement on these two mints and *not* on the real fields, which is the
    # right way round: the real fields would still be right.
    ".physics.pden_plasma_core_rad_mw_unclipped": (
        lambda d: d.physics.pden_plasma_core_rad_mw
    ),
    ".physics.pden_plasma_outer_rad_mw_unclipped": (
        lambda d: d.physics.pden_plasma_outer_rad_mw
    ),
    # --- the three entries that let the coil island come out of EXCLUDED_NODE_NAMES ---
    #
    # `.stellarator.wp_width_r_min` -- `Intersect`'s unknown, the raw crossing point of
    # `intersect(wp_width_r, lhs, wp_width_r, rhs, ...)`. A local in
    # `winding_pack_total_size` (`process/models/stellarator/coils/
    # calculate.py:452-465`), never stored. But `calculate.py:481,489` write
    # `awp_rad = wp_width_r_min` straight into the real field
    # `data.tfcoil.dr_tf_wp_with_insulation`, *after* the turn-size
    # clamp `wp_width_r_min = max(dx_tf_turn_general**2, wp_width_r_min)`
    # (`calculate.py:465`, ported at `coils/calculate.py:778`). So this identity is
    # exact whenever the clamp is inactive, and a lower bound otherwise. Measured on
    # this run: `dx_tf_turn_general**2 = 3.136e-03` against
    # `dr_tf_wp_with_insulation = 7.170e-01`, i.e. the clamp is inactive by 228x, so the
    # seed is exact here. It is in any case only a `RootFind`'s **starting guess** --
    # `Intersect` re-solves its own residual from it, and the solved answer is then
    # compared back against this same value like any other output. A clamped run would
    # start the Newton slightly low and still converge to the true crossing.
    ".stellarator.wp_width_r_min": (lambda d: d.tfcoil.dr_tf_wp_with_insulation),
    # `.tfcoil.a_tf_wp_no_insulation` / `.tfcoil.a_tf_wp_with_insulation` -- Python
    # locals in `winding_pack_total_size` (`calculate.py:494`/`:498`; the source's own
    # comment on `:493` says "(not global)"), never stored, but written *from* real
    # fields on those same lines:
    #
    #   a_tf_wp_no_insulation   = awp_tor * awp_rad
    #                           = .tfcoil.dx_tf_wp_primary_toroidal
    #                             * .tfcoil.dr_tf_wp_with_insulation
    #   a_tf_wp_with_insulation = (dr_tf_wp_with_insulation + 2*dx_tf_wp_insulation)
    #                             * (dx_tf_wp_primary_toroidal + 2*dx_tf_wp_insulation)
    #
    # -- `calculate.py:483-491` assign `awp_tor`/`awp_rad` to those two real fields
    # immediately before, so all three right-hand sides are stored values. Independently
    # confirmed on the converged run rather than read off the source alone: PROCESS's own
    # `data.tfcoil.j_tf_wp = coilcurrent * 1e6 / a_tf_wp_no_insulation`
    # (`calculate.py:499`) is reproduced to the last printed digit
    # (`3.0620392270788945e7`) by dividing `.tfcoil.c_tf_total / n_tf_coils` by the
    # reconstruction above.
    #
    # Not circular as a *comparison*: the right-hand sides are PROCESS's stored numbers,
    # so scoring the port's `a_tf_wp_*` against them checks that the port's own resolved
    # `wp_width_r_min` matches PROCESS's, which is exactly what `Intersect` is on the
    # hook for.
    ".tfcoil.a_tf_wp_no_insulation": (
        lambda d: d.tfcoil.dx_tf_wp_primary_toroidal * d.tfcoil.dr_tf_wp_with_insulation
    ),
    ".tfcoil.a_tf_wp_with_insulation": (
        lambda d: (
            (d.tfcoil.dr_tf_wp_with_insulation + 2.0 * d.tfcoil.dx_tf_wp_insulation)
            * (d.tfcoil.dx_tf_wp_primary_toroidal + 2.0 * d.tfcoil.dx_tf_wp_insulation)
        )
    ),
    # --- the two `.tokamak.build` mints (`_audit/units/models/build.md`) --------------
    #
    # Both are values PROCESS computes and then **overwrites in place**, so neither has a
    # `DataStructure` field of its own -- the same shape as the two `_unclipped`
    # radiation mints above, and the same resolution: each is reconstructible from stored
    # fields by an identity read straight off PROCESS's own source.
    #
    # Both are live now: `run_mda_harness --input <tokamak IN.DAT>` reaches them, and
    # `.tfcoil.dx_tf_wp_conductor_max` in particular is one edge of the
    # build/winding-pack cycle `mda.CUTS` closes. They were added with the nodes, before
    # that harness existed, because the reconstruction is a property of the port rather
    # than of the run -- and because an ungrounded mint is exactly what silently scores
    # against `0.0` the moment someone does point the harness at a tokamak.
    #
    # **Neither collides with the stellarator.** Every stellarator-only entry above is
    # ungrounded-but-unowned on the tokamak and vice versa, checked rather than assumed;
    # in particular the tokamak's winding-pack areas are
    # `.superconducting_tfcoil.a_tf_wp_*` where the stellarator's are
    # `.tfcoil.a_tf_wp_*`, so the two `a_tf_wp` identities above -- which encode the
    # *stellarator's* winding-pack geometry -- are never applied to a tokamak variable.
    #
    # `.tfcoil.dx_tf_wp_conductor_max` -- `process/models/build.py:1570-1572`, inside
    # `plasma_outboard_edge_toroidal_ripple`'s superconducting arm. Every operand is
    # stored. `1.2173980800120443` on `large_tokamak_eval.IN.DAT`.
    ".tfcoil.dx_tf_wp_conductor_max": (
        lambda d: (
            d.tfcoil.dx_tf_wp_primary_toroidal
            - 2.0 * (d.tfcoil.dx_tf_wp_insulation + d.tfcoil.dx_tf_wp_insertion_gap)
        )
    ),
    # `.build.r_tf_outboard_mid_unrippled` -- `process/models/build.py:1901-1909`, the
    # outboard stack-up *before* `:1939` may raise it to satisfy the ripple constraint.
    # Exact whenever the ripple constraint is inactive and a lower bound otherwise,
    # which is the same caveat `.stellarator.wp_width_r_min` carries and the same way
    # round: a run where the constraint bit would disagree on this mint and not on
    # `.build.r_tf_outboard_mid` itself. `13.988666666666669` on
    # `large_tokamak_eval.IN.DAT`.
    ".build.r_tf_outboard_mid_unrippled": (
        lambda d: (
            d.build.r_shld_outboard_outer
            + d.build.dr_shld_blkt_gap
            + d.build.dr_vv_outboard
            + d.build.gapomin
            + d.build.dr_shld_thermal_outboard
            + d.build.dr_tf_shld_gap
            + 0.5 * d.build.dr_tf_outboard
        )
    ),
}
"""`{ungrounded VarPath: data -> its exact analytic value}`, for cases where `0.0` is
not just ungrounded but actively wrong-*shaped*.

**Was `{".physics.profile_x": ...}`, now empty -- fixed at the source, not worked
around here.** `.physics.profile_x` (minted, read by `FusionRates`) turned out to be a
duplicate of `ProfileGrid`'s own `.physics.radius_plasma_profile_norm` (same
`linspace(0, 1, n_plasma_profile_elements)` formula, same real source) that this
harness's own comparison run surfaced -- `fusion_reactions.py`/`.md` were corrected to
read `radius_plasma_profile_norm` directly, so `.physics.profile_x` is no longer read
by anything (confirmed: `grep -rn "physics\\.profile_x" functional_process/models/`
finds no remaining `FromExactly`/`Output` binding, only historical docstring mentions). Kept
as an empty dict, not deleted outright, since a future ungrounded-and-wrong-shaped
case is exactly what this mechanism is for.
"""

DEVICE_ROOTS = ("stellarator", "tokamak")
"""The two device trees a machine's nodes can hang off, as they are spelled in a
`NodePath` (`.stellarator.coils.intersect`, `.tokamak.build.tf_outboard_mid`). Everything
else -- `.physics.*`, `.costs.*`, `.power.*` -- is device-agnostic and shared."""


def device_root(graph) -> str | None:
    """Which of `DEVICE_ROOTS` this graph's device-specific nodes sit under, or `None`
    for a graph (a `subgraph`, a test fixture) that has none.

    Read off the assembled graph rather than passed in, for the same reason
    `switch_audit` introspects instead of parsing: the caller that knows which `IN.DAT`
    it ran is not the caller that needs the answer, and a harness argument saying
    "stellarator" while the graph says otherwise is exactly the class of mistake this
    module exists to catch.

    Raises
    ------
    ValueError
        If nodes from both device trees are present. `machine_from_indat` builds one
        device, so this cannot happen today; it is the check that keeps the two
        machine-specific tables below meaningful if that ever changes.
    """
    found = {
        root
        for root in DEVICE_ROOTS
        for node in graph.nodes
        if node.path_str().startswith(f".{root}.")
    }
    if len(found) > 1:
        raise ValueError(
            f"this graph has nodes under both {sorted(found)} -- the "
            f"device-gated entries in KNOWN_UNVERIFIABLE_OUTPUTS cannot say which "
            f"PROCESS caller ran"
        )
    return next(iter(found), None)


STELLARATOR = "stellarator"
"""Guard value in `KNOWN_UNVERIFIABLE_OUTPUTS`: this entry applies only where
`device_root(graph)` says so. Spelled as the root's own name so the table reads as a
table and cannot drift from `DEVICE_ROOTS`."""

ANY_DEVICE = None
"""Guard value in `KNOWN_UNVERIFIABLE_OUTPUTS`: applies on every machine. No entry uses
it today -- all three known cases turned out to be one PROCESS caller discarding a value
another one stores -- but an output that no caller stores would take it."""

KNOWN_UNVERIFIABLE_OUTPUTS = {
    ".fwbs.f_a_fw_coolant_inboard": STELLARATOR,
    ".fwbs.f_a_fw_coolant_outboard": STELLARATOR,
    ".physics.fusrat": STELLARATOR,
}
"""`{VarPath: which device the exclusion applies on}` -- see `STELLARATOR`/`ANY_DEVICE`.

**Every entry is device-gated, and that is a finding, not a convenience.** All three were
written against the stellarator reference run and all three read, in this module's own
words, "PROCESS itself never actually writes a meaningful value there **for the arm this
port's node represents**". The arm is the whole content of the exclusion, and the tokamak
takes the other arm in all three cases -- checked against `process/` directly rather than
assumed from the port:

* `.fwbs.f_a_fw_coolant_*` are Python locals in `st_fwbs_s2`
  (`stellarator.py:672-678`, `:828`), and on the tokamak they are owned by a different
  node entirely (`.tokamak.ccfe_hcpb.first_wall_coolant_void_fractions` against
  `.stellarator.fwbs.blanket_shield_power`) modelling `hcpb.py:483,490`, which assigns
  **`self.data.fwbs.f_a_fw_coolant_inboard`/`_outboard`** -- real stored fields.
* `.physics.fusrat` is the third of `phyaux`'s seven returns, unpacked into a bare local
  `_fusrat` by the stellarator caller (`stellarator.py:2383`) and assigned to
  `self.data.physics.fusrat` by the tokamak one (`physics.py:961`). The port's node
  (`.physics.auxiliary_physics_quantities`) is the *same* node on both machines, which is
  why this one cannot be gated by owner and needs the device.

So an ungated set would have silently skipped three real checks on the tokamak. Left as
`STELLARATOR` rather than deleted because the stellarator half of each is still true:
against the Helias run these three still compare a correct formula to
`DataStructure()`'s uninitialised default.

The original evidence, unchanged: `VarPath`s where a real `DataStructure` field exists
(so `_ground_truth` succeeds, unlike `errors`' "no field at all" case) but PROCESS itself
never actually writes a meaningful value there for the arm this port's node represents --
comparing against whatever the field's uninitialised default happens to be is not a real
check, it is a guaranteed false positive.

The first two entries are `DetailedPowerflowBlanketShieldPower`'s own two "best-effort"
outputs -- its class docstring (`stellarator_fwbs_s2.py:378-382`) already says these
stay Python-locals in the real PROCESS source, never written to `data`, "matching
their PROCESS field names" only for naming convenience. Found via this harness's own
comparison run flagging `f_a_fw_coolant_inboard` as a "disagreement" (`got=0.444`,
`expected=0.0`) that traced back to exactly this documented, pre-existing caveat --
`expected=0.0` was `DataStructure()`'s bare default, not a PROCESS answer. The node's
other 14 outputs are real fields and stay in ordinary comparison scope; only these two
are excluded.

`.physics.fusrat` (`AuxiliaryPhysicsQuantities`) is the same case, confirmed against
the PROCESS source directly rather than inferred from the port: `phyaux` returns seven
values, and the stellarator caller unpacks the third into a bare local `_fusrat`
(`process/models/stellarator/stellarator.py:2383`) -- it is the *only* one of the seven
not assigned to a `self.data.physics.*` field there. The tokamak caller
(`process/models/physics/physics.py:961`) does store it, so the field is real, just
never written on this pipeline; `.physics.fusrat`'s declared default is `0.0`
(`process/data_structure/physics_variables.py:1730`), which is exactly the `expected`
the harness was comparing against. So the reported `got=1.059e21 vs expected=0.0` was
never a value disagreement -- it was the port's (correct) formula against an
uninitialised default. `physics_A_pure_formulas.md`'s data-footprint table (row
`phyaux` / `.physics.fusrat`) and `AuxiliaryPhysicsQuantities`'s own class docstring
(`physics_A_pure_formulas.py:378-381`) had already recorded the discard; nothing
downstream in the registered graph reads the field, so keeping the `Output` declared
(matching `physics.py`'s real field) and excluding it here is the same treatment the
two `f_a_fw_coolant_*` entries get.
"""


SWITCH = "switch"
"""Kind (a), `_audit/switch_elimination_design.md` §3: a model-selection switch. The
default classification, and the only one `switch_audit` treats as a switch. Every one of
these must be `IntEnum`-typed (`_audit/model_tree_design.md` §4); `SwitchAudit.
not_enum_typed` reports any that is not."""

SHAPE = "shape/resolution (kind b)"
"""Kind (b): an array shape or loop trip count. Static because `jit` needs it concrete,
not because anything is being chosen. Still resolved and value-checked against the run
where a same-named `DataStructure` field exists -- a wrong resolution is as much a bug as
a wrong switch -- but reported as what it is, not as a model choice."""

SET_MEMBERSHIP = "set membership (kind c)"
"""Kind (c): which members exist. A set, not a choice."""

ASSEMBLY_PAYLOAD = "assembly-time payload"
"""Neither a switch nor a shape: a value the graph carries that PROCESS never stores."""

FROZEN_INPUT = "frozen run input"
"""A real `DataStructure` **input** the graph carries statically, because something
derived from it was evaluated once at assembly time.

Distinct from `ASSEMBLY_PAYLOAD`, which PROCESS never stores: this one has a backing
field and is therefore value-checked against the converged run like any other kwarg. The
only thing it is not, is a *choice* -- it names no model.

The one instance is `TfCoilQuenchHeatCurrentDensity`'s `tftmp` /
`temp_tf_conductor_quench_max`, which fix the Gauss-Legendre grid the helium property
table was built at (`models/tfcoil/quench.py`). Freezing an input is only sound while it
cannot move, so `indat._quench_helium_table` refuses to assemble a machine whose `ixc`
names either -- the classification and that refusal are two halves of one claim.
"""

STATIC_KWARG_KINDS = {
    "n_plasma_profile_elements": SHAPE,
    "n_cs_pf_coils": SHAPE,
    "tftmp": FROZEN_INPUT,
    "temp_tf_conductor_quench_max": FROZEN_INPUT,
    "den_helium_at_nodes": ASSEMBLY_PAYLOAD,
    "cp_helium_at_nodes": ASSEMBLY_PAYLOAD,
    # `Avail2`/`AvailSt`'s pump counts. Not on the reference graph today (neither node
    # is registered on this configuration), classified anyway so that registering one
    # later does not have to remember to: the classification belongs with the kwarg,
    # not with whichever graph happens to carry it.
    "n_vac_pumps_high": SHAPE,
    "redun_vac": SHAPE,
    "imp_indices": SET_MEMBERSHIP,
    "machine_config": ASSEMBLY_PAYLOAD,
    "rho": ASSEMBLY_PAYLOAD,
}
"""`{static kwarg name: kind}` for every static kwarg that is **not** a model-selection
switch. Anything absent is kind (a), a switch.

`_audit/switch_elimination_design.md` §3's whole warning is that "switch" names four
different things and that conflating them is the main way this refactor goes wrong: a
document listing `n_plasma_profile_elements=201` as a "model" would be worse than what
exists now. So (b) and (c) are named here rather than left to pass as anonymous
integers through a line that says "static switch kwargs checked" -- the classification
is data, visible in the report, not a silent convention.

It is a *classification*, not an exemption. `n_plasma_profile_elements` and
`n_cs_pf_coils` still resolve to real `DataStructure` fields and are still compared
against the converged run exactly as before; they simply count as shape rather than as
choice. `imp_indices`/`machine_config`/`rho` have no backing field at all and are
carried by `STATIC_KWARGS_WITHOUT_BACKING_FIELD` below, which holds the evidence for
*why* -- this table holds only which of the four kinds each one is.
"""

_QUENCH_HELIUM_TABLE_REASON = (
    "helium density / isobaric specific heat at the 75 Gauss-Legendre nodes of "
    "`[tftmp, temp_tf_conductor_quench_max]` -- the whole CoolProp surface of the "
    "tokamak scope, evaluated once by `indat._quench_helium_table` at graph-assembly "
    "time. PROCESS asks `PropsSI` for the same numbers inside its own body and stores "
    "them nowhere, so there is no `DataStructure` field to check against; what *is* "
    "checked is the pair of temperatures that generate them, which are ordinary "
    "`FROZEN_INPUT` kwargs resolving to `.tfcoil.tftmp` and "
    "`.tfcoil.temp_tf_conductor_quench_max`. See "
    "`models/tfcoil/quench.py::TfCoilQuenchHeatCurrentDensity` for why the table is "
    "static at all and what refuses a machine on which it would go stale"
)

STATIC_KWARGS_WITHOUT_BACKING_FIELD = {
    "machine_config": (
        "the parsed contents of this machine's `stella_conf.json` -- a "
        "graph-assembly-time fact with no `DataStructure` field of its own (PROCESS "
        "reads the file and scatters its values straight into "
        "`.stellarator_config.*`, keeping the mapping nowhere). "
        "`StellaratorMachineConfig` owns those 34 fields, and *they* are checked by "
        "ordinary value comparison; the payload it selects them from is not a switch"
    ),
    "rho": (
        "the normalised radius the neoclassical profiles are sampled at -- PROCESS "
        "passes the literal `0.6` at `neoclassics.py:290` and stores it nowhere. The "
        "same-named field `.neoclassics.r_eff` is declared `= 0.0` and never assigned "
        "by anything in `process/`, so binding it as an `FromExactly` (which this node did "
        "until `_audit/boundary_inputs_audit.md` §6.1) evaluated every profile on axis"
    ),
    "imp_indices": (
        "which impurity species exist -- a graph-assembly-time fact with no "
        "`DataStructure` field of its own, see `ImpurityRadiationTotals`'s docstring "
        "and `total_process.py`'s own comment on its registration"
    ),
    "den_helium_at_nodes": _QUENCH_HELIUM_TABLE_REASON,
    "cp_helium_at_nodes": _QUENCH_HELIUM_TABLE_REASON,
}
"""Static kwargs confirmed *not* to be switches backed by a real `DataStructure` field,
with the reason. Anything static that is neither here nor resolvable by name is
reported as an unresolved entry, not silently dropped -- the same three-way discipline
`ComparisonReport` already applies to ungrounded/unverifiable/errors.

**There is no alias table any more.** One used to exist, for exactly one entry:
`PlasmaComposition.is_ignited` was a plain `bool` restating `i_plasma_ignited == 1`, so
name-based resolution could not reach the field it stood for and the mapping had to be
written out by hand as `(".physics.i_plasma_ignited", lambda v: bool(v == 1))`. That
kwarg is now spelled and typed as PROCESS spells it (`i_plasma_ignited`, typed
`PlasmaIgnitionModel`), which is `_audit/switch_elimination_design.md` §3(d)'s
prescription for kind-(d) aliases -- *delete, don't rename* -- and it resolves by name
like everything else. The entry did not move between the audit's categories: it was
counted in `checked` through the alias and is counted in `checked` by name now.
"""


@dataclass
class SwitchMismatch:
    """One node's static switch kwarg whose registered value differs from the value
    the converged run actually used.
    """

    node: str
    kwarg: str
    path: str
    registered: object
    actual: object
    enum: type | None = None
    """The `IntEnum` this kwarg is typed with, when it has one. Carried so the report
    can say `PROCESS_1990 != KOVARI_2014` instead of `0 != 1` -- which is the
    vocabulary the mistake was actually made in (`_audit/switch_elimination_design.md`
    §5(A): a name cannot be silently copied off the wrong default the way an integer
    can). `None` for kinds (b)/(c), which are numbers and sets and have no member names.
    """

    def spell_registered(self) -> str:
        """The registered value, with its enum member name where it has one."""
        return spell_switch_value(self.registered, self.enum)

    def spell_actual(self) -> str:
        """The converged run's value, with its enum member name where it has one."""
        return spell_switch_value(self.actual, self.enum)


@dataclass
class SwitchAudit:
    """Result of `switch_audit()` -- see `ComparisonReport.switches`."""

    checked: int = 0
    mismatches: list = field(default_factory=list)
    no_backing_field: list = field(default_factory=list)
    """`(node, kwarg, reason)` -- static kwargs that are genuinely not `data`-backed
    switches (`STATIC_KWARGS_WITHOUT_BACKING_FIELD`)."""
    unresolved: list = field(default_factory=list)
    """`(node, kwarg, why)` -- static kwargs this check could not map to a
    `DataStructure` field *and* that are not declared non-switches. These are the
    entries a future pass must triage; they are neither passes nor failures."""
    not_switches: list = field(default_factory=list)
    """`(node, kwarg, kind, disposition)` -- every static kwarg classified as something
    other than a model-selection switch by `STATIC_KWARG_KINDS`, with what the audit
    then did with it. Cuts across the three counted categories rather than replacing
    any of them: a kind-(b) shape that resolves to a real field is *both* `checked` and
    listed here. It exists so `switch_elimination_design.md` §3's "(b)/(c)/(d) must be
    explicitly reclassified so they stop being counted as switches at all" is something
    the report states, rather than something a reader has to know.
    """
    not_enum_typed: list = field(default_factory=list)
    """`(node, kwarg, type name)` -- kind-(a) switches whose registered value is a bare
    `int`/`bool` rather than an `IntEnum` member. **Must always be empty**:
    `_audit/model_tree_design.md` §4 makes enum typing the rule for every
    model-selection setting, and this is what keeps a new registration from quietly
    reintroducing the bare integer. Not a value failure -- the value may well be right
    -- so it is reported separately from `mismatches` and does not move the
    checked/mismatched/not-data-backed/unresolved line.
    """


def _declaration_modules(obj, seen):
    """Every `functional_process`-defined `equinox.Module` reachable from one graph
    node definition -- the declaration instance itself plus anything it holds.

    `Graph.definitions` holds `cottax.spec.CallableNode`s, not the declaration
    classes `total_process.py` instantiates; the declaration is reachable as
    `node.fn.__self__` (a bound method of the `NodalDeclaration` instance). Walking
    generically rather than special-casing that one shape also covers `Problem` nodes
    and any future wrapper. `equinox`'s own internal `Module`s (e.g. `BoundMethod`,
    which carries a `static` `__func__` field) are filtered out by module origin --
    they are not this project's registrations.

    Yields
    ------
    :
        Each reachable `functional_process` `equinox.Module`, in discovery order.
    """
    if id(obj) in seen:
        return
    seen.add(id(obj))
    if isinstance(obj, eqx.Module):
        if type(obj).__module__.startswith("functional_process"):
            yield obj
        for f in dataclasses.fields(obj):
            yield from _declaration_modules(getattr(obj, f.name, None), seen)
    elif isinstance(obj, (tuple, list)):
        for item in obj:
            yield from _declaration_modules(item, seen)
    elif callable(obj) and hasattr(obj, "__self__"):
        yield from _declaration_modules(obj.__self__, seen)


def _backing_field(data, name: str):
    """`(dotted path, live value)` for the `DataStructure` field named `name`, found by
    scanning every area for an attribute of exactly that name.

    PROCESS's own naming is what makes this work: field names are globally unique
    across areas in practice (`naming_convention.md`'s `<type>_<system>_<description>_
    <units>` scheme), so a bare kwarg name like `i_p_coolant_pumping` identifies one
    field. That is checked, not assumed -- a name found in two areas is returned as
    ambiguous rather than resolved to whichever came first.
    """
    hits = [
        area.name
        for area in dataclasses.fields(data)
        if hasattr(getattr(data, area.name), name)
    ]
    if len(hits) != 1:
        return None, hits
    return f".{hits[0]}.{name}", getattr(getattr(data, hits[0]), name)


def _switch_enum(declaration, f, registered) -> type | None:
    """The `IntEnum` a static kwarg is typed with, or `None`.

    Two sources, in order. The **registered value's own type** is the authority: it is
    what the assembled graph actually carries, which is the same "introspection, not
    source parsing" principle `switch_audit` is built on. Falling back to the
    **annotation** covers the case where a registration passed a bare `int` into an
    enum-typed slot -- exactly the mistake `SwitchAudit.not_enum_typed` reports, and
    the case where naming the enum in the report is most useful.
    """
    if isinstance(registered, IntEnum):
        return type(registered)
    annotation = f.type
    if isinstance(annotation, str):
        # `from __future__ import annotations` (or a quoted annotation) leaves the
        # string; resolve it against the declaring class's own module namespace.
        module = sys.modules.get(type(declaration).__module__)
        annotation = getattr(module, annotation, None) if module else None
    if isinstance(annotation, type) and issubclass(annotation, IntEnum):
        return annotation
    return None


def spell_switch_value(value, enum: type | None) -> str:
    """`"IGNITED (1)"` where `enum` names the value, `repr(value)` otherwise.

    Both halves on purpose: the member name is what a reader reasons about, the integer
    is what the `DataStructure` field literally holds and what a stale audit record or
    IN.DAT line will be spelled with. A value with no member in `enum` (a genuinely
    out-of-range integer, which is itself worth seeing) falls back to the bare `repr`.
    """
    if enum is None:
        return repr(value)
    try:
        return f"{enum(int(value)).name} ({int(value)})"
    except (ValueError, TypeError):
        return repr(value)


def _same_switch_value(registered, actual) -> bool:
    """Registered-vs-actual comparison that survives `IntEnum`/`numpy` scalars.

    Non-numeric switch values (`.vacuum.i_vacuum_pumping` is a `str` in PROCESS) fall
    back to plain equality.
    """
    if isinstance(registered, bool) or isinstance(actual, bool):
        return bool(registered) == bool(actual)
    try:
        return int(registered) == int(actual)
    except (TypeError, ValueError):
        return registered == actual


def switch_audit(graph, data) -> SwitchAudit:
    """Check every static switch kwarg registered on `graph`'s nodes against the value
    the converged run in `data` actually used.

    This closes a whole defect class at once. A node registration in
    `total_process.py` carries hardcoded static kwargs (`ConfinementTime(
    i_confinement_time=38, ...)`) and, historically, those values were copied from the
    corresponding `process/data_structure/*_variables.py` bare Python default rather
    than from the run being modelled. Four such bugs have been found one at a time, each
    only because a downstream value diverged loudly enough to notice
    (`i_confinement_time`, `i_thermal_electric_conversion`, `i_p_coolant_pumping`,
    `i_plasma_ignited` -- `next_steps.md` §8). Nothing checked them directly until this
    function; a switch whose wrong value happens not to move any compared output would
    never have been caught at all.

    Introspection, not source parsing: the kwargs are `eqx.field(static=True)`
    attributes on the assembled graph's own declaration instances (see
    `confinement_time.py:2006-2008`), so what is checked is what the graph actually
    carries, not what `total_process.py`'s text says.

    **Enum-aware, and kind-aware.** The mechanism above is unchanged -- same walk, same
    name resolution, same numeric comparison -- but two things are now reported rather
    than left implicit. A kwarg typed with an `IntEnum` (`_audit/model_tree_design.md`
    §4) has both sides of a mismatch spelled with member names, because
    `PROCESS_1990 != KOVARI_2014` is the form in which the mistake is legible and
    `0 != 1` is not. And a kwarg that is not a model-selection switch at all --
    `_audit/switch_elimination_design.md` §3's kind (b) shapes and kind (c) sets -- is
    recorded as such in `not_switches`, so the report never presents an array length as
    a model choice. Kind (d), the `bool` alias, no longer exists to classify: it was
    deleted rather than renamed, which is what §3(d) prescribes.
    """
    audit = SwitchAudit()
    seen_pairs = set()
    for node_path, node in graph.definitions.items():
        node_name = node_path.path_str()
        for declaration in _declaration_modules(node, set()):
            for f in dataclasses.fields(declaration):
                if not f.metadata.get("static"):
                    continue
                key = (node_name, f.name)
                if key in seen_pairs:
                    continue
                seen_pairs.add(key)
                registered = getattr(declaration, f.name, None)
                kind = STATIC_KWARG_KINDS.get(f.name, SWITCH)
                enum = _switch_enum(declaration, f, registered)
                if kind is SWITCH and not isinstance(registered, IntEnum):
                    audit.not_enum_typed.append((
                        node_name,
                        f.name,
                        type(registered).__name__,
                    ))
                reason = STATIC_KWARGS_WITHOUT_BACKING_FIELD.get(f.name)
                if reason is not None:
                    audit.no_backing_field.append((node_name, f.name, reason))
                    if kind is not SWITCH:
                        audit.not_switches.append((
                            node_name,
                            f.name,
                            kind,
                            "no backing DataStructure field",
                        ))
                    continue
                path, live = _backing_field(data, f.name)
                if path is None:
                    audit.unresolved.append((
                        node_name,
                        f.name,
                        "no DataStructure field of this name"
                        if not live
                        else f"name exists in several areas: {live}",
                    ))
                    if kind is not SWITCH:
                        audit.not_switches.append((
                            node_name,
                            f.name,
                            kind,
                            "unresolved",
                        ))
                    continue
                actual = live
                audit.checked += 1
                if kind is not SWITCH:
                    audit.not_switches.append((
                        node_name,
                        f.name,
                        kind,
                        f"value-checked against {path}",
                    ))
                if not _same_switch_value(registered, actual):
                    audit.mismatches.append(
                        SwitchMismatch(
                            node=node_name,
                            kwarg=f.name,
                            path=path,
                            registered=registered,
                            actual=actual,
                            enum=enum,
                        )
                    )
    return audit


EXPLAINED_DISAGREEMENTS = {
    ".vacuum.dia_vv_vacuum_ducts": (
        "`VacuumOld`'s duct diameter, ~2.9e-4 high. **Not a floating-point path "
        "difference -- a deliberate, already-documented solver-tolerance difference.** "
        "This port solves the duct-diameter equation to a relative-step tolerance of "
        "`1e-10` (`functional_process/models/vacuum.py:250`'s "
        "`solve_duct_diameter(..., tol=1e-10)`, whose own docstring at lines 262-271 "
        "states the deviation and why); PROCESS stops the *same* Newton iteration at "
        "`dd <= 0.01`, a 1% relative-step cutoff "
        "(`process/models/vacuum.py:469-477`). PROCESS's own stopping rule therefore "
        "admits ~1e-2 relative error in this field, ~34x *larger* than the observed "
        "difference -- PROCESS's number is not ground truth here, the same Tier2 "
        "framing `_audit/next_steps.md` §8 already records for `Intersect`/"
        "`DuctDiameterRootFind`. A second, independently documented deviation points "
        "the same way: PROCESS tests its fits-in-the-gap condition on the diameter "
        "*before* the Newton update, this port on the diameter actually returned "
        "(`functional_process/models/vacuum.py:391-397`)."
    ),
    ".heat_transport.p_plant_electric_base_total_mw": (
        "**Not a port defect: PROCESS's own converged `DataStructure` is internally "
        "inconsistent here, and the port is the self-consistent side.** Measured "
        "directly by instrumenting `Buildings.run` across a real solve: "
        "`Buildings.run(output=False)` (the solve pass) leaves "
        "`.buildings.a_plant_floor_effective = 563075.16`, and "
        "`Buildings.run(output=True)` (the final report pass) leaves `680433.44` -- the "
        "*same method*, differing only because `Stellarator.run(output=True)` reruns "
        "`st_build`/`st_coil` in the opposite order to the solve pass, so "
        "`.build.z_tf_inside_half` is `4.1556` at solve time and `7.3592` in the report "
        "(the dual-write `next_steps.md` §8 records as `ZTfInsideHalf`), and "
        "`Buildings.run` reads it as `tf_vertical_dim` (`process/models/buildings."
        "py:52-54`). `Stellarator.run(output=True)` then calls "
        "`power.output_plant_electric_powers()` rather than "
        "`plant_electric_production()` (`stellarator.py:148-152` against `:169-172`), so "
        "`p_plant_electric_base_total_mw` is **never recomputed** in the report pass and "
        "keeps its solve-pass value while `a_plant_floor_effective` moves. The port "
        "models the reported arm throughout, so it computes "
        "`5 + 680433.44 * 1.5e-4 = 107.065` from PROCESS's own formula and PROCESS's own "
        "stored inputs; PROCESS's stored `89.461` corresponds to the *other* "
        "`a_plant_floor_effective`. **Every one of the eighteen disagreements in this "
        "chain is that single `+17.604 MW` offset propagated linearly** -- checked "
        "arithmetically, not asserted: `p_plant_core_systems_elec_mw`, "
        "`p_plant_secondary_heat_mw`, `p_plant_electric_recirc_mw` and `Acpow.tlvpmw` "
        "each differ by exactly `17.604`, `p_plant_electric_net_mw` by exactly "
        "`-17.604`, and the rest is that delta through the linear cost accumulation to "
        "`.costs.coe` (`rel_diff = 1.73e-2`)."
    ),
    ".pf_coil.n_pf_coil_turns": (
        "**Not a value defect: the disagreement lives entirely in the array's dead "
        "tail.** `PFCoilSizes` owns the 22-slot array whole and writes a structural "
        "`0.0` at every index past the plasma circuit (8-21), where PROCESS's "
        "converged `DataStructure` still holds `100.0` -- the residue of "
        "`pfcoil.py:605-608`'s `first_call` bootstrap (`n_pf_coil_turns[:] = 100.0`), "
        "which nothing ever overwrites at indices no coil occupies. The eight live "
        "entries (coils 0-6 and the plasma circuit) agree to tier-1 precision; "
        "`14/22 off, worst [8]` in the report is exactly the dead tail. The same "
        "residue appears once more as the `^hat.pf_coil.n_pf_coil_turns` minted "
        "unknown of the PF coil cycle's FixedPoint -- one cause, two rows. Not "
        "filtered, per this table's own rule: a *live*-index disagreement on this "
        "array must still surface."
    ),
    ".vacuum.dlscal": (
        "Not independent of `.vacuum.dia_vv_vacuum_ducts` above -- the *same* error, "
        "propagated. `dlscal = l1*d**1.4 + (ltot - l1)*(1.2*d)**1.4` "
        "(`process/models/vacuum.py:424`), so a relative error `e` in `d` becomes "
        "`1.4*e` in `dlscal`. Measured: `d` is off by 2.941e-04 and `dlscal` by "
        "4.118e-04; `1.4 * 2.941e-04 = 4.117e-04`, agreeing to four digits. That "
        "arithmetic is the confirmation that both `VacuumOld` disagreements are one "
        "cause, not two, and that the cause is the duct-diameter solve."
    ),
}
"""`VarPath`s whose disagreement has been traced to a *documented, deliberate*
algorithmic difference rather than a porting defect.

**Deliberately not wired into `compare()`.** These stay ordinary, fully-counted
disagreements in the report. Suppressing them would need either a per-field tolerance
or another exclusion set, and either would mask a genuine future regression on the same
field -- if `dlscal` ever moves to 4%, that must still show up. This is documentation
for the reader who finds `VacuumOld` in the "all disagreements" list and needs to know
it has already been chased, not a filter.

**Measured on the stellarator reference run, and only the two `VacuumOld` entries
generalise.** Because nothing here filters anything, an entry that does not apply to the
machine in hand costs nothing and is simply never looked at -- but a reader should know
which is which. The `VacuumOld` pair is about a shared subsystem and a solver tolerance
this port sets in its own source, so it says the same thing on any machine that registers
`VacuumOld`. `.heat_transport.p_plant_electric_base_total_mw`'s explanation is
**stellarator-specific in its entirety**: it rests on `Stellarator.run(output=True)`
re-running `st_build`/`st_coil` in the opposite order to the solve pass, and on
`stellarator.py:148-152` calling `output_plant_electric_powers()` where the tokamak calls
`plant_electric_production()`. A tokamak disagreement on that same field would be a
different finding wearing the same name and must be chased from scratch.
"""


_CACHE_VERSION = "v2"
"""Bumped when `_cache_key`'s *definition* changes, so old entries can never be read
back under a key whose meaning has moved.

`v1` hashed **every** `*IN.DAT` and `*.json` in the input file's directory rather than
the named file's own. All nine input files under `tests/regression/input_files/`
therefore shared one key, and `converged_data("large_tokamak_eval.IN.DAT")` returned
whichever run had been cached first -- in practice the stellarator's converged state,
with `rmajor = 26.69` and `n_tf_coils = 50` where the tokamak has ~8.9 and 16. A cache
that returns another run's answer is worse than no cache at all, and no test could see
it while only one device was being ported.
"""


def _cache_key(input_file: str) -> str:
    """A digest of everything a converged run depends on: **this run's own** input files
    (the named `.IN.DAT` and its `.stella_conf.json` companion, if it has one) and the
    state of the `process/` source tree.

    Only this run's inputs, deliberately -- not every file beside them, and not every
    file in the directory. PROCESS writes `OUT.DAT`/`MFILE.DAT`/`SIG_TF.json` into that
    same directory, and `SingleRun.__init__` creates some of them before this is ever
    called, so hashing the directory wholesale makes the key depend on whether anything
    has run there yet; hashing every *input* in it makes every run in the directory
    share one key, which is the `v1` bug `_CACHE_VERSION` records.

    The sidecar is found by PROCESS's own `output_prefix` convention -- the stem before
    `.IN.DAT`, plus `.stella_conf.json` -- which is what `Stellarator.st_new_config()`
    opens and what `indat.REFERENCE_STELLA_CONF` names. A run without one simply has
    nothing to add; its absence is hashed as absence, so creating one later is a miss.

    The source fingerprint is `(relative path, size, mtime_ns)` per `.py` file, not
    their contents -- ~600 `stat` calls instead of ~10 MB of hashing, and it changes on
    every edit, which is the property that matters. A checkout that restores an old
    mtime would collide; nothing here does that.
    """
    import hashlib

    h = hashlib.sha256()
    h.update(_CACHE_VERSION.encode())
    named = Path(input_file).resolve()
    h.update(named.name.encode())
    h.update(named.read_bytes())
    # PROCESS's `output_prefix` convention: `<stem>.IN.DAT` -> `<stem>.stella_conf.json`.
    stem = (
        named.name[: -len(".IN.DAT")] if named.name.endswith(".IN.DAT") else named.stem
    )
    sidecar = named.parent / f"{stem}.stella_conf.json"
    h.update(b"stella_conf:")
    if sidecar.is_file():
        h.update(sidecar.name.encode())
        h.update(sidecar.read_bytes())
    else:
        h.update(b"<absent>")
    import process as _process

    root = Path(_process.__file__).parent
    for path in sorted(root.rglob("*.py")):
        stat = path.stat()
        h.update(str(path.relative_to(root)).encode())
        h.update(f"{stat.st_size}:{stat.st_mtime_ns}".encode())
    return h.hexdigest()[:32]


CACHE_DIR = Path(
    os.environ.get("FP_HARNESS_CACHE_DIR", Path.home() / ".cache/functional_process")
)
"""Where converged `DataStructure`s are cached. Override with `FP_HARNESS_CACHE_DIR`;
disable the cache entirely with `FP_HARNESS_NO_CACHE=1`."""


def converged_data(input_file: str, use_cache: bool = True):
    """Run PROCESS's own `SingleRun` on `input_file` to convergence, in-process, and
    return the resulting live `DataStructure`. Writes the usual `OUT.DAT`/`MFILE.DAT`
    beside `input_file` as a side effect (same as any real PROCESS run) -- not
    suppressed, since nothing here depends on it not happening.

    **Cached on disk**, because this solve is ~95 s and dominates every harness run:
    changing one line of `functional_process/` and re-measuring should cost the graph
    evaluation, not another full PROCESS solve. The key covers the input files *and*
    the `process/` source tree (see `_cache_key`), so editing the reference
    implementation invalidates it -- which is the case that would otherwise produce a
    silently stale "expected" column, the single worst failure mode a harness can have.
    Pass `use_cache=False`, or set `FP_HARNESS_NO_CACHE=1`, to force the solve.
    """
    from process.main import SingleRun

    use_cache = use_cache and not os.environ.get("FP_HARNESS_NO_CACHE")
    cached = CACHE_DIR / f"converged-{_cache_key(input_file)}.pkl" if use_cache else None
    if cached is not None and cached.exists():
        with cached.open("rb") as f:
            return pickle.load(f)  # noqa: S301 -- our own file, written just below

    run = SingleRun(input_file, "vmcon")
    run.run()
    if cached is not None:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        # Write-then-rename: an interrupted run leaves no half-written cache entry to
        # be read back as a converged solve.
        partial = cached.with_suffix(".partial")
        with partial.open("wb") as f:
            pickle.dump(run.data, f)
        partial.replace(cached)
    return run.data


def _without_excluded(graph):
    to_delete = tuple(
        n
        for n in graph.nodes
        if any(name in n.path_str() for name in EXCLUDED_NODE_NAMES)
    )
    if not to_delete:
        return graph
    return Delete(to_delete).apply(graph)


@dataclass
class Disagreement:
    """One `VarPath` where the schedule's answer and `data`'s own converged value
    don't agree within tolerance.

    For an array-valued variable, `got`/`expected` are the **worst single element**
    (largest relative difference), and `shape`/`index`/`n_off` say which one and how
    many of its siblings are also off -- so one array is one disagreement, not one per
    element, and the printed number is still a number a reader can chase.
    """

    var: VarPath
    owner: NodePath
    got: float
    expected: float
    shape: tuple | None = None
    """`None` for a scalar; the array's shape otherwise."""
    index: tuple | None = None
    """`None` for a scalar; the multi-index of the worst element otherwise."""
    n_off: int | None = None
    """`None` for a scalar; how many elements are outside tolerance otherwise."""

    @property
    def rel_diff(self) -> float:
        """`|got - expected| / |expected|` (or `/1` at `expected == 0`)."""
        denom = abs(self.expected) if self.expected != 0 else 1.0
        return abs(self.got - self.expected) / denom

    @property
    def where(self) -> str:
        """`""` for a scalar; ` [shape=(201,) worst [37], 12/201 off]` otherwise."""
        if self.shape is None:
            return ""
        size = int(np.prod(self.shape)) if self.shape else 1
        return f" [shape={self.shape} worst {list(self.index)}, {self.n_off}/{size} off]"


@dataclass
class ComparisonReport:
    """The result of one `compare()` call -- see each field's own docstring."""

    agreements: int = 0
    disagreements: list = field(default_factory=list)
    driven_block_disagreements: list = field(default_factory=list)
    acyclic_disagreements: list = field(default_factory=list)
    errors: list = field(default_factory=list)
    ungrounded_inputs: list = field(default_factory=list)
    """Boundary inputs / driven unknowns with no real `DataStructure` field to seed
    from -- the same shape as `DuctDiameterRootFind`'s exclusion (a minted local
    PROCESS never stores), just discovered empirically rather than pre-flagged.
    Seeded `0.0` (a scalar placeholder -- an array-shaped ungrounded input will still
    break shape-dependent downstream arithmetic, see this module's own known-issues
    note) so the schedule can still run; anything downstream is reported separately,
    in `unverifiable`, not scored as a pass or a fail.
    """
    unverifiable: list = field(default_factory=list)
    """Outputs that read an `ungrounded_inputs` entry, directly or transitively --
    not a disagreement, since there is nothing real to disagree with.
    """
    switches: SwitchAudit = field(default_factory=SwitchAudit)
    """`switch_audit()`'s result -- every registered static switch kwarg checked
    against the value this run actually used. Independent of the value comparison
    below: a wrong switch can be caught here even when it moves no compared output.
    """
    owned_total: int = 0
    """How many owned variables `compare()` walked. Every one of them must land in
    exactly one of `agreements`/`disagreements`/`unverifiable`/`errors` -- see
    `unaccounted`.
    """
    unaccounted: list = field(default_factory=list)
    """Owned variables that landed in no bucket at all. **Must always be empty.**
    It exists because it once was not: array-valued outputs were dropped by a bare
    `continue` inside the float conversion, so 29 of 487 owned variables were scored
    as neither pass nor fail and nothing said so (`_audit/boundary_inputs_audit.md`
    §6.2). The invariant is cheap; the hole it closes cost a real wrong answer
    (`ProfileValues.rho`) its visibility for as long as it was open.
    """
    array_agreements: int = 0
    """How many of `agreements` were array-valued (compared elementwise). Reported
    separately only so a change in array handling is visible in the summary.
    """
    trivial_agreements: list = field(default_factory=list)
    """Agreements where the port and `data` are **both exactly zero** everywhere.

    They are real agreements and stay counted in `agreements` (the accounting
    invariant is over buckets, not over how much each bucket proves), but they say
    "this path is switched off in this configuration", not "the port reproduces
    PROCESS" -- no arithmetic in the node was exercised. Reported separately so the
    headline agreement count cannot be read as more coverage than it is.

    Named because `next_steps.md` §11.6 item 6's `atol` hole -- real, and closed by
    `compare`'s `atol=0.0` -- turned out not to be the only vacuous agreement here,
    and this is the larger of the two: 73 of the reference run's 499.
    """

    def summary(self) -> str:
        lines = [
            f"agreements: {self.agreements} "
            f"(of which array-valued: {self.array_agreements}, "
            f"both-sides-exactly-zero: {len(self.trivial_agreements)})",
            f"disagreements: {len(self.disagreements)}",
            f"  in driven (cyclic) blocks: {len(self.driven_block_disagreements)}",
            f"  in ordinary acyclic nodes: {len(self.acyclic_disagreements)}",
            f"unverifiable (depends on an ungrounded input): {len(self.unverifiable)}",
            f"ungrounded inputs (no real DataStructure field): "
            f"{len(self.ungrounded_inputs)}",
            f"errors (could not evaluate at all): {len(self.errors)}",
            f"owned variables walked: {self.owned_total}, unaccounted: "
            f"{len(self.unaccounted)}"
            + ("" if not self.unaccounted else "  <-- MUST BE 0"),
            f"static switch kwargs checked: {self.switches.checked}, "
            f"mismatched: {len(self.switches.mismatches)}, "
            f"not data-backed: {len(self.switches.no_backing_field)}, "
            f"unresolved: {len(self.switches.unresolved)}",
        ]
        if self.switches.mismatches:
            lines.append("\nstatic switch mismatches (registered vs. this run):")
            for m in self.switches.mismatches:
                lines.append(
                    f"  {m.node}.{m.kwarg}: "
                    f"{m.spell_registered()} != {m.spell_actual()} "
                    f"(registered vs. {m.path})"
                )
        if self.switches.not_enum_typed:
            lines.append(
                "\nstatic switch kwargs NOT IntEnum-typed  <-- MUST BE EMPTY "
                "(model_tree_design.md §4):"
            )
            for node, kwarg, type_name in self.switches.not_enum_typed:
                lines.append(f"  {node}.{kwarg}: registered as bare {type_name}")
        if self.switches.not_switches:
            lines.append(
                "\nstatic kwargs classified as NOT model-selection switches "
                "(switch_elimination_design.md §3):"
            )
            for node, kwarg, kind, disposition in self.switches.not_switches:
                lines.append(f"  {node}.{kwarg}: {kind} -- {disposition}")
        if self.switches.unresolved:
            lines.append("\nstatic kwargs neither checked nor declared non-switches:")
            for node, kwarg, why in self.switches.unresolved:
                lines.append(f"  {node}.{kwarg}: {why}")
        if self.switches.no_backing_field:
            lines.append("\nstatic kwargs with no backing DataStructure field:")
            for node, kwarg, reason in self.switches.no_backing_field:
                lines.append(f"  {node}.{kwarg}: {reason}")
        if self.ungrounded_inputs:
            lines.append("\nungrounded inputs:")
            for v in self.ungrounded_inputs:
                lines.append(f"  {v.path_str()}")
        by_owner: dict = {}
        for d in self.disagreements:
            by_owner.setdefault(d.owner.path_str(), []).append(d)
        worst = sorted(by_owner.items(), key=lambda kv: -max(d.rel_diff for d in kv[1]))
        lines.append("\nworst offenders by node:")
        for owner, ds in worst[:20]:
            worst_d = max(ds, key=lambda d: d.rel_diff)
            lines.append(
                f"  {owner}: {len(ds)} var(s) off, worst "
                f"{worst_d.var.path_str()} got={worst_d.got!r} "
                f"expected={worst_d.expected!r} rel_diff={worst_d.rel_diff:.3e}"
                f"{worst_d.where}"
            )
        # Every disagreement, not just each node's worst. The per-node summary above
        # reports only one variable per node, which is what hid
        # `AuxiliaryPhysicsQuantities`'s second off variable from a reader who had
        # only the summary -- a node's largest rel_diff is not always its most
        # diagnostic one, and two off variables on one node need not share a cause.
        if self.disagreements:
            lines.append("\nall disagreements:")
            for owner, ds in worst:
                for d in sorted(ds, key=lambda d: -d.rel_diff):
                    lines.append(
                        f"  {owner} {d.var.path_str()}: got={d.got!r} "
                        f"expected={d.expected!r} rel_diff={d.rel_diff:.3e}"
                        f"{d.where}"
                    )
        if self.errors:
            # Every error, not the first 20. The cap silently hid errors 21+ from a
            # reader who had only the summary -- both machines currently sit at 20-25,
            # and the count line above already says how many there are, so a listing
            # that stopped short contradicted its own header. A very long list still
            # says what it dropped rather than dropping it silently.
            lines.append("\nerrors:")
            for e in self.errors[:100]:
                lines.append(f"  {e}")
            if len(self.errors) > 100:
                lines.append(f"  ... and {len(self.errors) - 100} more")
        return "\n".join(lines)


def _ground_truth(data, var: VarPath):
    """`data`'s own value at `var`, resolved in order:

    1. `KNOWN_MINT_VALUES` -- an exact analytic value for a mint with no real field.
    2. `cottax.tools.minting.unminted` -- if `var` is a `FixedPointCut`'s own minted
       unknown (e.g. `^hat.physics.proton_rate_density`), the real place it names. A
       minted copy has no `DataStructure` field of its own; at the fixed point it
       equals the real variable it was cut from, which *does* --
       `FixedPointFunction`'s structural self-loops are the opposite way round (the
       problem owns the real var, reads the minted copy), so this only matters for
       the two `FixedPointCut`s this module adds, not the 8 pre-existing structural
       ones.
    3. `var` itself.
    """
    from cottax.tools.minting import unminted
    from cottax.tools.pytree import get_at

    known = KNOWN_MINT_VALUES.get(var.path_str())
    if known is not None:
        return known(data)
    return get_at(data, unminted(var).keys)


def _diff(var: VarPath, owner: NodePath, got, expected, *, rtol, atol):
    """Compare one owned variable's computed value against `data`'s own.

    Returns `None` if they agree, a `Disagreement` if they don't, and a **string** if
    the pair cannot be compared at all (non-numeric, or shapes that don't match) --
    three outcomes and no fourth, because the fourth used to be a silent `continue`.

    Arrays are compared elementwise and reported as one disagreement carrying their
    worst element, so an off-by-one profile is one line in the report rather than 201.
    """
    try:
        got_a = np.asarray(got, dtype=float)
        expected_a = np.asarray(expected, dtype=float)
    except (TypeError, ValueError) as e:
        return f"not numeric, cannot compare {var.path_str()} (owned by {owner}): {e}"
    if got_a.shape != expected_a.shape:
        return (
            f"shape mismatch for {var.path_str()} (owned by {owner}): port "
            f"{got_a.shape} vs data {expected_a.shape}"
        )
    close = np.isclose(got_a, expected_a, rtol=rtol, atol=atol, equal_nan=True)
    if close.all():
        return None
    if got_a.ndim == 0:
        return Disagreement(
            var=var, owner=owner, got=float(got_a), expected=float(expected_a)
        )
    denom = np.where(expected_a == 0.0, 1.0, np.abs(expected_a))
    rel = np.abs(got_a - expected_a) / denom
    # `nan` sorts last under `argmax`, and a `nan` element is exactly the one worth
    # reporting, so name it explicitly rather than letting it lose to a finite cell.
    bad = ~np.isfinite(rel)
    flat = int(np.argmax(np.where(bad, np.inf, rel)) if bad.any() else np.argmax(rel))
    index = np.unravel_index(flat, got_a.shape)
    return Disagreement(
        var=var,
        owner=owner,
        got=float(got_a[index]),
        expected=float(expected_a[index]),
        shape=tuple(got_a.shape),
        index=tuple(int(i) for i in index),
        n_off=int((~close).sum()),
    )


def _is_trivially_zero(got, expected) -> bool:
    """Is this agreeing pair zero on **both** sides, everywhere?

    Split out of `compare`'s classification only so it is directly testable -- the
    category it names is `ComparisonReport.trivial_agreements`, and the reason it is
    worth naming is in that field's docstring.
    """
    return (
        not np.asarray(expected, dtype=float).any()
        and not np.asarray(got, dtype=float).any()
    )


def compare(graph, data, rtol=1e-6, atol=0.0) -> ComparisonReport:
    """Drive `graph` (any `total_process`-shaped `Graph`) from `data`'s own converged
    values, and diff every value the schedule produces against `data` itself.

    **`atol` is 0.0 -- the comparison is purely relative.** It used to be `1e-9`, on
    the reasoning that a variable small enough should not be held to `rtol`. An
    absolute floor cannot serve a `DataStructure` whose fields span 1e-15 to 1e+20:
    it silently exempts whole subsystems chosen by their units.

    On the code as it stands the floor is **inert** -- of 499 agreements, 0 depend on
    it -- which is not the argument for removing it, because it is inert only while
    nothing is wrong down there. Measured with a known bug reintroduced
    (`ProfileValues.rho` set back to `0.0`, the old `r_eff` binding's value):
    `.neoclassics.temperatures` comes out `1.9997e-15` against PROCESS's `1.1705e-15`
    and `.neoclassics.dr_temperatures` comes out `-0.0` against `-1.215e-15` -- **71 %
    and 100 % wrong, and `atol=1e-9` reports both as agreements.** At `atol=0.0` both
    are disagreements, and the harness sees 4 of that bug's fields rather than 2.
    That is what the floor costs, stated as a defect it actually hides rather than as
    a defect it might.

    A pair that is genuinely zero on both sides still agrees (`|0 - 0| <= rtol * 0`)
    and is counted in `trivial_agreements`.
    """
    report = ComparisonReport()
    # Audited on the graph *as passed in*, before `_without_excluded`: the excluded
    # nodes are minted islands with nothing to compare numerically, but their static
    # switch kwargs are registered values like any other and are just as capable of
    # being wrong. (`i_tf_sc_mat` is no longer among them: it selects an occupant of
    # `winding_pack_intersect_inputs` now, and a slot's answer is checked by
    # `test_machine.py`, not by this audit.)
    report.switches = switch_audit(graph, data)

    graph = _without_excluded(graph)
    driven = driven_graph(graph)
    blocking = Blocking.scc(driven)
    # Drivers live in the graph now (`Assign`, applied by `driven_graph`), so
    # `schedule_for` takes none.
    schedule = schedule_for(blocking)

    env = {}
    ungrounded = []
    # `^guess.*` ports are unowned inputs like any other, but there is nothing in
    # `data` spelled that way -- they are seeded below, from the unknown each one
    # starts. Grounding them here would report every one as ungrounded.
    starts = {
        guess
        for problem, problem_type in zip(
            blocking.problems, blocking.problem_types, strict=True
        )
        if problem_type is not None
        for _, guess in starts_for(driven, problem)
    }
    for var in driven.unowned_inputs:
        if var in starts:
            continue
        try:
            env[var] = jnp.asarray(_ground_truth(data, var))
        except (AttributeError, KeyError):
            ungrounded.append(var)
            env[var] = jnp.asarray(0.0)  # placeholder -- see `ungrounded_inputs`
    # Starting guesses for every driven unknown -- same converged run's own value,
    # written to the unknown's `^guess.*` port.
    for problem, problem_type in zip(
        blocking.problems, blocking.problem_types, strict=True
    ):
        if problem_type is None:
            continue
        for var, guess in starts_for(driven, problem):
            try:
                env[guess] = jnp.asarray(_ground_truth(data, var))
            except (AttributeError, KeyError):
                ungrounded.append(var)
                env[guess] = jnp.asarray(0.0)

    report.ungrounded_inputs = ungrounded
    # Everything that reads an ungrounded input, directly or transitively, can only
    # ever disagree with `data` because its own input was a placeholder, not because
    # the port is wrong -- excluded from pass/fail, reported separately.
    unverifiable_owners = set()
    for var in ungrounded:
        for reader in driven.readers.get(var, ()):
            unverifiable_owners.add(reader)
            unverifiable_owners |= set(driven.descendants([reader]))

    try:
        out = schedule(env)
    except Exception as e:  # noqa: BLE001 -- report, don't crash the harness
        report.errors.append(f"schedule() raised: {type(e).__name__}: {e}")
        return report

    # `KNOWN_UNVERIFIABLE_OUTPUTS`' entries are gated on which PROCESS caller ran, so
    # the device is resolved once, off the graph itself, rather than per variable or
    # from an argument the caller could get wrong (`device_root`'s own docstring).
    device = device_root(driven)
    unverifiable_here = {
        path
        for path, only_on in KNOWN_UNVERIFIABLE_OUTPUTS.items()
        if only_on is ANY_DEVICE or only_on == device
    }
    for var, owner in driven.owners.items():
        report.owned_total += 1
        if owner in unverifiable_owners or var.path_str() in unverifiable_here:
            report.unverifiable.append(var)
            continue
        try:
            expected = _ground_truth(data, var)
        except (AttributeError, KeyError) as e:
            report.errors.append(f"no DataStructure field for {var.path_str()}: {e}")
            continue
        if var not in out:
            report.errors.append(f"schedule did not produce {var.path_str()}")
            continue
        d = _diff(var, owner, out[var], expected, rtol=rtol, atol=atol)
        if isinstance(d, str):  # not comparable at all -- say so, never drop it
            report.errors.append(d)
        elif d is None:
            report.agreements += 1
            if np.asarray(expected).ndim:
                report.array_agreements += 1
            # Both conversions already succeeded inside `_diff` (it returns a string
            # otherwise), so neither `asarray` can raise here.
            if _is_trivially_zero(out[var], expected):
                report.trivial_agreements.append(var)
        else:
            report.disagreements.append(d)
            block_index = blocking.index[owner]
            if blocking.problem_types[block_index] is not None:
                report.driven_block_disagreements.append(d)
            else:
                report.acyclic_disagreements.append(d)

    accounted = (
        report.agreements
        + len(report.disagreements)
        + len(report.unverifiable)
        + len(report.errors)
    )
    if accounted != report.owned_total:
        # Not raised: a harness that refuses to report because its own bookkeeping is
        # off is worse than one that reports and says so. The summary flags it.
        report.unaccounted = [f"{report.owned_total - accounted} owned variable(s)"]

    return report
