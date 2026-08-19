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
from dataclasses import dataclass, field

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from cottax.blocking import Blocking
from cottax.evaluate import schedule_for
from cottax.plan import Delete
from cottax.spec import NodePath, VarPath

from functional_process.mda import default_drivers, driven_graph

EXCLUDED_NODE_NAMES = ("DuctDiameterRootFind",)
"""`DuctDiameterRootFind`: see this module's own docstring -- no real `DataStructure`
field backs any of its `VarPath`s.

**The coil island (`Intersect`/`WindingPackIntersectInputs`/`WindingPackTotalSizePost`)
used to be here too, and is not any more** -- see
`_audit/constraint_32_investigation.md` for the full evidence. The exclusion rested on
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
finds no remaining `Input`/`Output` binding, only historical docstring mentions). Kept
as an empty dict, not deleted outright, since a future ungrounded-and-wrong-shaped
case is exactly what this mechanism is for.
"""

KNOWN_UNVERIFIABLE_OUTPUTS = frozenset({
    ".fwbs.f_a_fw_coolant_inboard",
    ".fwbs.f_a_fw_coolant_outboard",
    ".physics.fusrat",
})
"""`VarPath`s where a real `DataStructure` field exists (so `_ground_truth` succeeds,
unlike `errors`' "no field at all" case) but PROCESS itself never actually writes a
meaningful value there for the arm this port's node represents -- comparing against
whatever the field's uninitialised default happens to be is not a real check, it is a
guaranteed false positive.

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


STATIC_KWARG_ALIASES = {
    "is_ignited": (
        ".physics.i_plasma_ignited",
        lambda v: bool(v == 1),
    ),
}
"""`{static kwarg name: (backing dotted path, data value -> expected kwarg value)}`,
for the one case where a node's static kwarg is *not* spelled like the PROCESS field
it stands for.

`PlasmaComposition.is_ignited` (`physics_B_composition.py:418`) is a plain `bool`, not
the raw `int` switch: its body is `if is_ignited:` (`physics_B_composition.py:219`),
and its own docstring (`physics_B_composition.py:134-136`) records that it stands for
PROCESS's `PlasmaIgnitionModel(i_plasma_ignited) == PlasmaIgnitionModel.NON_IGNITED`
compare -- i.e. it is `i_plasma_ignited == 1` (`IGNITED`,
`physics_variables.py:45-49`). Name-based resolution cannot recover that mapping, so
it is declared here rather than silently reported as "no backing field".
"""

STATIC_KWARGS_WITHOUT_BACKING_FIELD = {
    "imp_indices": (
        "which impurity species exist -- a graph-assembly-time fact with no "
        "`DataStructure` field of its own, see `ImpurityRadiationTotals`'s docstring "
        "and `total_process.py`'s own comment on its registration"
    ),
}
"""Static kwargs confirmed *not* to be switches backed by a real `DataStructure` field,
with the reason. Anything static that is neither here nor resolvable by name (nor in
`STATIC_KWARG_ALIASES`) is reported as an unresolved entry, not silently dropped --
the same three-way discipline `ComparisonReport` already applies to
ungrounded/unverifiable/errors.
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
    `confinement_time.py:1986-1988`), so what is checked is what the graph actually
    carries, not what `total_process.py`'s text says.
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
                reason = STATIC_KWARGS_WITHOUT_BACKING_FIELD.get(f.name)
                if reason is not None:
                    audit.no_backing_field.append((node_name, f.name, reason))
                    continue
                alias = STATIC_KWARG_ALIASES.get(f.name)
                if alias is not None:
                    path, convert = alias
                    _, _, field_name = path.rpartition(".")
                    resolved, live = _backing_field(data, field_name)
                    if resolved is None:
                        audit.unresolved.append((
                            node_name,
                            f.name,
                            f"alias target {path} resolved to areas {live}",
                        ))
                        continue
                    actual = convert(live)
                else:
                    path, live = _backing_field(data, f.name)
                    if path is None:
                        audit.unresolved.append((
                            node_name,
                            f.name,
                            "no DataStructure field of this name"
                            if not live
                            else f"name exists in several areas: {live}",
                        ))
                        continue
                    actual = live
                audit.checked += 1
                if not _same_switch_value(registered, actual):
                    audit.mismatches.append(
                        SwitchMismatch(
                            node=node_name,
                            kwarg=f.name,
                            path=path,
                            registered=registered,
                            actual=actual,
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
"""


def converged_data(input_file: str):
    """Run PROCESS's own `SingleRun` on `input_file` to convergence, in-process, and
    return the resulting live `DataStructure`. Writes the usual `OUT.DAT`/`MFILE.DAT`
    beside `input_file` as a side effect (same as any real PROCESS run) -- not
    suppressed, since nothing here depends on it not happening.
    """
    from process.main import SingleRun

    run = SingleRun(input_file, "vmcon")
    run.run()
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
    don't agree within tolerance."""

    var: VarPath
    owner: NodePath
    got: float
    expected: float

    @property
    def rel_diff(self) -> float:
        """`|got - expected| / |expected|` (or `/1` at `expected == 0`)."""
        denom = abs(self.expected) if self.expected != 0 else 1.0
        return abs(self.got - self.expected) / denom


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

    def summary(self) -> str:
        lines = [
            f"agreements: {self.agreements}",
            f"disagreements: {len(self.disagreements)}",
            f"  in driven (cyclic) blocks: {len(self.driven_block_disagreements)}",
            f"  in ordinary acyclic nodes: {len(self.acyclic_disagreements)}",
            f"unverifiable (depends on an ungrounded input): {len(self.unverifiable)}",
            f"ungrounded inputs (no real DataStructure field): "
            f"{len(self.ungrounded_inputs)}",
            f"errors (could not evaluate at all): {len(self.errors)}",
            f"static switch kwargs checked: {self.switches.checked}, "
            f"mismatched: {len(self.switches.mismatches)}, "
            f"not data-backed: {len(self.switches.no_backing_field)}, "
            f"unresolved: {len(self.switches.unresolved)}",
        ]
        if self.switches.mismatches:
            lines.append("\nstatic switch mismatches (registered vs. this run):")
            for m in self.switches.mismatches:
                lines.append(
                    f"  {m.node}.{m.kwarg}: registered={m.registered!r} "
                    f"but {m.path} == {m.actual!r}"
                )
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
                    )
        if self.errors:
            lines.append("\nerrors:")
            for e in self.errors[:20]:
                lines.append(f"  {e}")
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


def compare(graph, data, rtol=1e-6, atol=1e-9) -> ComparisonReport:
    """Drive `graph` (any `total_process`-shaped `Graph`) from `data`'s own converged
    values, and diff every value the schedule produces against `data` itself.
    """
    report = ComparisonReport()
    # Audited on the graph *as passed in*, before `_without_excluded`: the excluded
    # nodes are minted islands with nothing to compare numerically, but their static
    # switch kwargs (e.g. `WindingPackIntersectInputs(i_tf_sc_mat=1)`) are registered
    # values like any other and are just as capable of being wrong.
    report.switches = switch_audit(graph, data)

    graph = _without_excluded(graph)
    driven = driven_graph(graph)
    blocking = Blocking.scc(driven)
    schedule = schedule_for(blocking, default_drivers(blocking))

    env = {}
    ungrounded = []
    for var in driven.unowned_inputs:
        try:
            env[var] = jnp.asarray(_ground_truth(data, var))
        except (AttributeError, KeyError):
            ungrounded.append(var)
            env[var] = jnp.asarray(0.0)  # placeholder -- see `ungrounded_inputs`
    # Starting guesses for every driven unknown -- same converged run's own value.
    for problem, problem_type in zip(
        blocking.problems, blocking.problem_types, strict=True
    ):
        if problem_type is None:
            continue
        for var in driven[problem].owns:
            try:
                env[var] = jnp.asarray(_ground_truth(data, var))
            except (AttributeError, KeyError):
                ungrounded.append(var)
                env[var] = jnp.asarray(0.0)

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

    for var, owner in driven.owners.items():
        if owner in unverifiable_owners or var.path_str() in KNOWN_UNVERIFIABLE_OUTPUTS:
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
        got = out[var]
        try:
            got_f = float(np.asarray(got))
            expected_f = float(np.asarray(expected))
        except (TypeError, ValueError):
            continue  # non-scalar or non-numeric field, skip
        if np.isclose(got_f, expected_f, rtol=rtol, atol=atol, equal_nan=True):
            report.agreements += 1
        else:
            d = Disagreement(var=var, owner=owner, got=got_f, expected=expected_f)
            report.disagreements.append(d)
            block_index = blocking.index[owner]
            if blocking.problem_types[block_index] is not None:
                report.driven_block_disagreements.append(d)
            else:
                report.acyclic_disagreements.append(d)

    return report
