"""Pure-functional port of `process/models/physics/current_drive.py`'s `CurrentDrive`.

Audit record: `functional_process/_audit/units/models/physics/current_drive.md`. Read it
first -- especially "A live PROCESS bug in two sibling arms" (`calculate_profile_y`
returns `None`) and "What this unit does *not* port and why", which between them say
exactly which of the eleven `i_hcd_primary` values this file can answer.

**Scope: the minimal closure that produces `_audit/tokamak_boundary.md`'s three
`.tokamak.current_drive` boundary reads** --
`.current_drive.p_hcd_ecrh_injected_total_mw`, `.current_drive.p_hcd_injected_total_mw`
and `.heat_transport.p_hcd_electric_total_mw` -- for the combination
`large_tokamak_eval.IN.DAT` actually holds: `i_hcd_primary = 10`
(`USER_INPUT_ELECTRON_CYCLOTRON`, the file's line 124), `i_hcd_secondary = 0`
(`NO_CURRENT_DRIVE`, PROCESS's default at `current_drive_variables.py:206` -- the file
never sets it), `i_hcd_calculations = 1` (default, `:223`) and `i_plasma_ignited = 0`
(`NON_IGNITED`, `physics_variables.py:881`). Every other heating-and-current-drive
scheme is UNPORTED; see the audit record for the per-value reason.

`CurrentDrive.current_drive` (`process/models/physics/current_drive.py:1651-2309`) is one
660-line method in which **four** switches interleave -- `i_hcd_calculations` gates the
whole body, `i_hcd_primary` picks both an efficiency formula and a wall-plug block,
`i_hcd_secondary` picks the same two things again for the second system, and
`i_plasma_ignited` decides whether the wall-plug total survives. Splitting it is the same
move `confinement_time.py` made: one class per switch value this port supports, each
declaring only the reads its own arm makes, so the graph stops claiming edges the run
does not have. Two of the seven stages below turn out to be switch-*independent* pure
algebra once the arms are peeled off, and they are single nodes.

`i_plasma_ignited` is a switch (`_audit/naming_convention.md` § "switches are not
ports"): a plain Python int used for ordinary branching in the composite
`calculate_current_drive_ecrh_primary_no_secondary` below, never traced. The harness
marks it `static_argnames`.
"""

from cottax.interfaces.pytree_namespace_module import (
    ExplicitFunction,
    From,
    OutputInto,
)

from functional_process.paths import current_drive, heat_transport, physics
from process.data_structure.physics_variables import PlasmaIgnitionModel

# ---------------------------------------------------------------------------
# Stage 1 -- the current drive efficiency of the primary heating system.
# `i_hcd_primary` selects one of eleven formulas; `hcd_models` in the source
# (`current_drive.py:1697-1771`) is a dict of eleven lambdas, of which exactly one is
# ever called. Each reads a different set of variables, which is why each is its own
# function here rather than one function behind a switch argument.
# ---------------------------------------------------------------------------


def user_input_electron_cyclotron_efficiency(
    *, eta_cd_norm_ecrh, nd_plasma_electrons_vol_avg, rmajor
):
    """ECRH current drive efficiency from a user-supplied normalised gamma.

    Ports `hcd_models[10]`, `process/models/physics/current_drive.py:1744-1747`, plus the
    `dene20` conversion its enclosing scope computes once at `:1690`.
    `CurrentDriveModel.USER_INPUT_ELECTRON_CYCLOTRON` (10), the only `i_hcd_primary`
    value this port answers.

    Three reads, and that is the whole model -- no plasma profile, no Coulomb logarithm,
    no local temperature. That is worth stating because it is *why* this value is the one
    ported first: the ten other arms of `hcd_models` reach `culecd`/`cullhy`/`culnbi`/
    `iternb` and through them the profile machinery, and two of those are broken in
    PROCESS itself (see the audit record).
    """
    dene20 = nd_plasma_electrons_vol_avg * 1.0e-20
    return eta_cd_norm_ecrh / (dene20 * rmajor)


# ---------------------------------------------------------------------------
# Stage 2 -- the secondary heating system's contribution, `i_hcd_secondary`.
# ---------------------------------------------------------------------------


def hcd_secondary_driven_current(
    *, eta_cd_hcd_secondary, p_hcd_secondary_injected_mw, plasma_current
):
    """Current driven by the secondary system, and its share of the plasma current.

    Ports `process/models/physics/current_drive.py:1821-1831`, unchanged. **Switch
    independent**: the source computes these two lines outside every `if`, from
    `eta_cd_hcd_secondary` -- which is what the `i_hcd_secondary` arms decide -- so once
    that efficiency is a variable the arms have already produced, this is ordinary
    algebra with no arm of its own.

    Returns
    -------
    tuple
        `(c_hcd_secondary_driven, f_c_plasma_hcd_secondary)`.
    """
    c_hcd_secondary_driven = eta_cd_hcd_secondary * p_hcd_secondary_injected_mw * 1.0e6
    return c_hcd_secondary_driven, c_hcd_secondary_driven / plasma_current


# ---------------------------------------------------------------------------
# Stage 3 -- the primary system's injected power. Switch independent.
# ---------------------------------------------------------------------------


def hcd_primary_injected_power_mw(
    *,
    f_c_plasma_auxiliary,
    f_c_plasma_hcd_secondary,
    plasma_current,
    eta_cd_hcd_primary,
):
    """Injected power the primary system needs to drive the auxiliary current fraction.

    Ports `process/models/physics/current_drive.py:1834-1842`, unchanged. The primary
    system is the *residual*: it supplies whatever share of the auxiliary current
    fraction the secondary did not, which is why this reads
    `f_c_plasma_hcd_secondary` and not the secondary's power directly.
    """
    return (
        1.0e-6
        * (f_c_plasma_auxiliary - f_c_plasma_hcd_secondary)
        * plasma_current
        / eta_cd_hcd_primary
    )


# ---------------------------------------------------------------------------
# Stage 4 -- the primary system's wall-plug block. Selected by the primary model's
# *method* (`CurrentDriveModel.method`), not by its value: `current_drive.py:2131`
# branches on `primary_cdm.method == ELECTRON_CYCLOTRON`, so one block serves all four
# ECRH models (3, 7, 10, 13). The ECRH injected-power accumulator it feeds is also
# written by the *secondary* ECRH block (`:1955`), which is why the occupant below is
# keyed on the pair and not on `i_hcd_primary` alone.
# ---------------------------------------------------------------------------


def electron_cyclotron_primary_powers(
    *,
    p_hcd_ecrh_injected_secondary_mw,
    p_hcd_primary_injected_mw,
    p_hcd_primary_extra_heat_mw,
    eta_ecrh_injector_wall_plug,
):
    """The ECRH primary block: wall-plug power and the ECRH injected-power total.

    Ports `process/models/physics/current_drive.py:2131-2156`. The source forms the sum
    `p_hcd_primary_injected_mw + p_hcd_primary_extra_heat_mw` three separate times
    (`:2132`, `:2139`, `:2148`) from the same two unchanged values; it is computed once
    here, which is the same number in float64 and not an approximation.

    `p_hcd_ecrh_injected_secondary_mw` is what the secondary system already added to
    `.current_drive.p_hcd_ecrh_injected_total_mw` before this block's `+=` at `:2147`
    (the accumulator is zeroed at `:1663`). It has no `DataStructure` field of its own --
    PROCESS accumulates in place -- so it is a plain argument, and the only occupant
    written below passes `0.0` for it because its arm's secondary system is
    `NO_CURRENT_DRIVE`.

    Returns
    -------
    tuple
        `(p_hcd_ecrh_injected_total_mw, p_hcd_ecrh_electric_mw,
        eta_hcd_primary_injector_wall_plug, p_hcd_primary_electric_mw)`.
    """
    p_hcd_primary_total_mw = p_hcd_primary_injected_mw + p_hcd_primary_extra_heat_mw

    p_hcd_primary_electric_mw = p_hcd_primary_total_mw / eta_ecrh_injector_wall_plug
    p_hcd_ecrh_injected_total_mw = (
        p_hcd_ecrh_injected_secondary_mw + p_hcd_primary_total_mw
    )
    p_hcd_ecrh_electric_mw = p_hcd_ecrh_injected_total_mw / eta_ecrh_injector_wall_plug
    return (
        p_hcd_ecrh_injected_total_mw,
        p_hcd_ecrh_electric_mw,
        eta_ecrh_injector_wall_plug,
        p_hcd_primary_electric_mw,
    )


# ---------------------------------------------------------------------------
# Stage 5 and 6 -- the totals. Switch independent given the arms' own outputs, except
# for `i_plasma_ignited`, which decides the wall-plug total outright.
# ---------------------------------------------------------------------------


def hcd_injected_power_total_mw(
    *,
    p_hcd_primary_injected_mw,
    p_hcd_primary_extra_heat_mw,
    p_hcd_secondary_injected_mw,
    p_hcd_secondary_extra_heat_mw,
):
    """Total injected power that contributed to heating.

    Ports `process/models/physics/current_drive.py:2265-2270`, unchanged. This is the
    variable `_audit/tokamak_boundary.md` § "What blocked the real file" names as the one
    the NON_IGNITED confinement head also reads -- see
    `confinement_time.py::PlasmaPowerLossNonIgnitedCoreRadiation`.
    """
    return (
        p_hcd_primary_injected_mw
        + p_hcd_primary_extra_heat_mw
        + p_hcd_secondary_injected_mw
        + p_hcd_secondary_extra_heat_mw
    )


def hcd_electric_total_mw(
    *, p_hcd_primary_electric_mw, p_hcd_secondary_electric_mw, i_plasma_ignited
):
    """Total wall plug power for all heating systems.

    Ports `process/models/physics/current_drive.py:2289-2299`, including the
    `i_plasma_ignited` reset the source's own comment calls a *"fudge"*: on an ignited
    plasma the whole electrical total is discarded and replaced by zero, so the two
    reads above are not reads at all on that arm. That is exactly the invented edge the
    occupant split exists to remove, and here it removes two -- see
    `HcdElectricTotalIgnited`, which reads nothing.
    """
    if PlasmaIgnitionModel(int(i_plasma_ignited)) == PlasmaIgnitionModel.IGNITED:
        return 0.0
    return p_hcd_primary_electric_mw + p_hcd_secondary_electric_mw


# ---------------------------------------------------------------------------
# The composite -- one function reproducing the whole of `CurrentDrive.current_drive`
# for the arm this port supports, so there is a boundary PROCESS itself has to diff
# against. `TestCurrentDriveEcrhPrimaryNoSecondary` calls it against a real
# `CurrentDrive` bound to a `DataStructure`, sample by sample, values and gradients.
# The node split below is finer than anything PROCESS exposes; this is where the two
# meet, the same trade `confinement_time.py::plasma_power_loss_mw` records.
# ---------------------------------------------------------------------------


def calculate_current_drive_ecrh_primary_no_secondary(
    *,
    eta_cd_norm_ecrh,
    nd_plasma_electrons_vol_avg,
    rmajor,
    plasma_current,
    f_c_plasma_auxiliary,
    p_hcd_primary_extra_heat_mw,
    p_hcd_secondary_injected_mw,
    eta_ecrh_injector_wall_plug,
    i_plasma_ignited,
):
    """`CurrentDrive.current_drive` for `i_hcd_primary = 10`, `i_hcd_secondary = 0`.

    Ports `process/models/physics/current_drive.py:1651-2309` restricted to the arm
    `large_tokamak_eval.IN.DAT` selects, with `i_hcd_calculations = 1`. Written as a
    composition of the stage functions above so there is exactly one source of truth for
    each formula and the node occupants cannot drift from what this is diffed against.

    Three of the source's writes are **not** recomputed here because on this arm PROCESS
    does not compute them either -- `.current_drive.eta_cd_hcd_secondary`,
    `.current_drive.p_hcd_secondary_extra_heat_mw` (forced to `0.0` at `:1682`) and
    `.heat_transport.p_hcd_secondary_electric_mw` are all zero, and
    `HcdSecondaryHeatingNone` below is the node that says so. They appear here as
    literals, and the audit record's data footprint table carries the evidence for each.

    Returns
    -------
    tuple
        `(eta_cd_hcd_primary, c_hcd_secondary_driven, f_c_plasma_hcd_secondary,
        p_hcd_primary_injected_mw, p_hcd_ecrh_injected_total_mw, p_hcd_ecrh_electric_mw,
        eta_hcd_primary_injector_wall_plug, p_hcd_primary_electric_mw,
        p_hcd_injected_total_mw, p_hcd_electric_total_mw)`.
    """
    eta_cd_hcd_primary = user_input_electron_cyclotron_efficiency(
        eta_cd_norm_ecrh=eta_cd_norm_ecrh,
        nd_plasma_electrons_vol_avg=nd_plasma_electrons_vol_avg,
        rmajor=rmajor,
    )

    c_hcd_secondary_driven, f_c_plasma_hcd_secondary = hcd_secondary_driven_current(
        eta_cd_hcd_secondary=0.0,
        p_hcd_secondary_injected_mw=p_hcd_secondary_injected_mw,
        plasma_current=plasma_current,
    )

    p_hcd_primary_injected_mw = hcd_primary_injected_power_mw(
        f_c_plasma_auxiliary=f_c_plasma_auxiliary,
        f_c_plasma_hcd_secondary=f_c_plasma_hcd_secondary,
        plasma_current=plasma_current,
        eta_cd_hcd_primary=eta_cd_hcd_primary,
    )

    (
        p_hcd_ecrh_injected_total_mw,
        p_hcd_ecrh_electric_mw,
        eta_hcd_primary_injector_wall_plug,
        p_hcd_primary_electric_mw,
    ) = electron_cyclotron_primary_powers(
        p_hcd_ecrh_injected_secondary_mw=0.0,
        p_hcd_primary_injected_mw=p_hcd_primary_injected_mw,
        p_hcd_primary_extra_heat_mw=p_hcd_primary_extra_heat_mw,
        eta_ecrh_injector_wall_plug=eta_ecrh_injector_wall_plug,
    )

    p_hcd_injected_total_mw = hcd_injected_power_total_mw(
        p_hcd_primary_injected_mw=p_hcd_primary_injected_mw,
        p_hcd_primary_extra_heat_mw=p_hcd_primary_extra_heat_mw,
        p_hcd_secondary_injected_mw=p_hcd_secondary_injected_mw,
        p_hcd_secondary_extra_heat_mw=0.0,
    )

    p_hcd_electric_total_mw = hcd_electric_total_mw(
        p_hcd_primary_electric_mw=p_hcd_primary_electric_mw,
        p_hcd_secondary_electric_mw=0.0,
        i_plasma_ignited=i_plasma_ignited,
    )

    return (
        eta_cd_hcd_primary,
        c_hcd_secondary_driven,
        f_c_plasma_hcd_secondary,
        p_hcd_primary_injected_mw,
        p_hcd_ecrh_injected_total_mw,
        p_hcd_ecrh_electric_mw,
        eta_hcd_primary_injector_wall_plug,
        p_hcd_primary_electric_mw,
        p_hcd_injected_total_mw,
        p_hcd_electric_total_mw,
    )


# ---------------------------------------------------------------------------
# cottax nodes
# ---------------------------------------------------------------------------


class HcdPrimaryEfficiency(ExplicitFunction):
    """The family that owns `.current_drive.eta_cd_hcd_primary`: one occupant per model.

    This is what `hcd_models` was: eleven lambdas in a dict, indexed by
    `i_hcd_primary`, of which one is called (`current_drive.py:1795-1798`). Each reads a
    different set -- model 10 reads three variables, model 13 reads six plus two more
    switches, models 6/7/8 reach the plasma profile machinery entirely -- so declaring
    the union would be the invented-edge defect at its widest in this file.
    """


class HcdPrimaryEfficiencyUserInputEcrh(HcdPrimaryEfficiency):
    """`i_hcd_primary == 10` (`USER_INPUT_ELECTRON_CYCLOTRON`).

    `large_tokamak_eval.IN.DAT:124`'s value, and the only one this port answers.
    """

    eta_cd_hcd_primary = OutputInto(current_drive)

    def __call__(
        self,
        eta_cd_norm_ecrh=From(current_drive),
        nd_plasma_electrons_vol_avg=From(physics),
        rmajor=From(physics),
    ):
        return user_input_electron_cyclotron_efficiency(
            eta_cd_norm_ecrh=eta_cd_norm_ecrh,
            nd_plasma_electrons_vol_avg=nd_plasma_electrons_vol_avg,
            rmajor=rmajor,
        )


class HcdSecondaryHeating(ExplicitFunction):
    """The family that owns what the *secondary* heating system contributes.

    `i_hcd_secondary` chooses among the same eleven efficiency models plus a twelfth
    value, `0`, meaning there is no secondary system at all.
    """


class HcdSecondaryHeatingNone(HcdSecondaryHeating):
    """`i_hcd_secondary == 0` (`NO_CURRENT_DRIVE`): the secondary contributes zero.

    PROCESS's default (`current_drive_variables.py:206`) and
    `large_tokamak_eval.IN.DAT`'s value, since that file never sets the switch.

    **A node with no reads, and that is the finding, not an accident.** Of the three
    fields it owns, PROCESS explicitly assigns only one -- `p_hcd_secondary_extra_heat_mw
    = 0.0` at `current_drive.py:1682`, guarded by `if i_hcd_secondary == 0` and by
    nothing else. The other two it simply never writes on this arm:
    `eta_cd_hcd_secondary` is skipped because `0` is not a key of `hcd_models`
    (`:1784-1787`) and `p_hcd_secondary_electric_mw` because every block that assigns it
    is guarded on a `secondary_cdm.method` that `NO_CURRENT_DRIVE` does not have. Both
    therefore hold their `DataStructure` defaults -- `0.0` at
    `current_drive_variables.py:98` and `heat_transport_variables.py:127` -- for the
    whole run, and no other model writes either (checked across `process/`).

    Declaring the zeros is what keeps two computed quantities off the boundary. The
    alternative is to leave them as inputs nothing produces, which would be a boundary
    entry standing for "PROCESS did not run this code", and `_audit/tokamak_boundary.md`
    § "The twelve that are simply inputs" is explicit that the boundary is for variables
    PROCESS *computes nowhere*, not for ones a switch happened to skip.
    """

    eta_cd_hcd_secondary = OutputInto(current_drive)
    p_hcd_secondary_extra_heat_mw = OutputInto(current_drive)
    p_hcd_secondary_electric_mw = OutputInto(heat_transport)

    def __call__(self):
        return 0.0, 0.0, 0.0


class HcdSecondaryDrivenCurrent(ExplicitFunction):
    """cottax node: `hcd_secondary_driven_current`, ports declared.

    Switch independent.
    """

    c_hcd_secondary_driven = OutputInto(current_drive)
    f_c_plasma_hcd_secondary = OutputInto(current_drive)

    def __call__(
        self,
        eta_cd_hcd_secondary=From(current_drive),
        p_hcd_secondary_injected_mw=From(current_drive),
        plasma_current=From(physics),
    ):
        return hcd_secondary_driven_current(
            eta_cd_hcd_secondary=eta_cd_hcd_secondary,
            p_hcd_secondary_injected_mw=p_hcd_secondary_injected_mw,
            plasma_current=plasma_current,
        )


class HcdPrimaryInjectedPower(ExplicitFunction):
    """cottax node: `hcd_primary_injected_power_mw`, ports declared.

    Switch independent.
    """

    p_hcd_primary_injected_mw = OutputInto(current_drive)

    def __call__(
        self,
        f_c_plasma_auxiliary=From(physics),
        f_c_plasma_hcd_secondary=From(current_drive),
        plasma_current=From(physics),
        eta_cd_hcd_primary=From(current_drive),
    ):
        return hcd_primary_injected_power_mw(
            f_c_plasma_auxiliary=f_c_plasma_auxiliary,
            f_c_plasma_hcd_secondary=f_c_plasma_hcd_secondary,
            plasma_current=plasma_current,
            eta_cd_hcd_primary=eta_cd_hcd_primary,
        )


class HcdPrimaryPowers(ExplicitFunction):
    """The family that owns the primary system's wall-plug and per-technology powers.

    Two switches decide this one, which is why its occupants are named for a pair the
    way `PlasmaPowerLossIgnitedCoreRadiation` is. `i_hcd_primary`'s *method* picks the
    block (`current_drive.py:2068`/`2099`/`2131`/`2162`/`2191`, one per technology), and
    `i_hcd_secondary`'s method decides how much that technology's injected-power
    accumulator already held when the block's `+=` ran -- the accumulators are zeroed at
    `:1663-1667` and the secondary blocks at `:1885-2063` add to them first. A
    per-technology "secondary contribution" variable would let the two be independent
    nodes, but PROCESS has no such field: it accumulates in place, so the pair is the
    honest key until someone adds one.
    """


class HcdPrimaryPowersElectronCyclotronNoSecondary(HcdPrimaryPowers):
    """Primary method `ELECTRON_CYCLOTRON` (`i_hcd_primary` 3, 7, 10, 13), secondary 0.

    Only `i_hcd_primary == 10` is reachable end to end in this port, because it is the
    only value with an `HcdPrimaryEfficiency` occupant -- but the block itself is keyed
    on the method, so the class is stated for the four values that share it rather than
    narrowed to the one whose upstream happens to be written.

    Reads `eta_ecrh_injector_wall_plug` and **not** the four sibling injector
    efficiencies (`eta_lowhyb_*`, `eta_icrh_*`, `eta_ebw_*`, `eta_beam_*`) that a node
    covering all five methods would have had to declare, nor `e_beam_kev`,
    `f_p_beam_orbit_loss`, `f_p_beam_shine_through` or `f_p_beam_injected_ions`, which
    only the neutral-beam block at `:2191-2260` reads.
    """

    p_hcd_ecrh_injected_total_mw = OutputInto(current_drive)
    p_hcd_ecrh_electric_mw = OutputInto(current_drive)
    eta_hcd_primary_injector_wall_plug = OutputInto(current_drive)
    p_hcd_primary_electric_mw = OutputInto(heat_transport)

    def __call__(
        self,
        p_hcd_primary_injected_mw=From(current_drive),
        p_hcd_primary_extra_heat_mw=From(current_drive),
        eta_ecrh_injector_wall_plug=From(current_drive),
    ):
        return electron_cyclotron_primary_powers(
            p_hcd_ecrh_injected_secondary_mw=0.0,
            p_hcd_primary_injected_mw=p_hcd_primary_injected_mw,
            p_hcd_primary_extra_heat_mw=p_hcd_primary_extra_heat_mw,
            eta_ecrh_injector_wall_plug=eta_ecrh_injector_wall_plug,
        )


class HcdInjectedPowerTotal(ExplicitFunction):
    """cottax node: `hcd_injected_power_total_mw`, ports declared. Switch independent."""

    p_hcd_injected_total_mw = OutputInto(current_drive)

    def __call__(
        self,
        p_hcd_primary_injected_mw=From(current_drive),
        p_hcd_primary_extra_heat_mw=From(current_drive),
        p_hcd_secondary_injected_mw=From(current_drive),
        p_hcd_secondary_extra_heat_mw=From(current_drive),
    ):
        return hcd_injected_power_total_mw(
            p_hcd_primary_injected_mw=p_hcd_primary_injected_mw,
            p_hcd_primary_extra_heat_mw=p_hcd_primary_extra_heat_mw,
            p_hcd_secondary_injected_mw=p_hcd_secondary_injected_mw,
            p_hcd_secondary_extra_heat_mw=p_hcd_secondary_extra_heat_mw,
        )


class HcdElectricTotal(ExplicitFunction):
    """The family that owns `.heat_transport.p_hcd_electric_total_mw`.

    `i_plasma_ignited` decides it, and the two arms could hardly differ more: one is the
    sum of two reads, the other is the literal `0.0` and reads nothing at all.
    """


class HcdElectricTotalNonIgnited(HcdElectricTotal):
    """`i_plasma_ignited == 0` (`NON_IGNITED`): the sum of the two systems' wall plugs.

    PROCESS's default (`physics_variables.py:881`) and `large_tokamak_eval.IN.DAT`'s
    value, since that file never sets the switch -- the same discovery
    `_audit/tokamak_boundary.md` § "What blocked the real file" makes about the
    confinement head.
    """

    p_hcd_electric_total_mw = OutputInto(heat_transport)

    def __call__(
        self,
        p_hcd_primary_electric_mw=From(heat_transport),
        p_hcd_secondary_electric_mw=From(heat_transport),
    ):
        return hcd_electric_total_mw(
            p_hcd_primary_electric_mw=p_hcd_primary_electric_mw,
            p_hcd_secondary_electric_mw=p_hcd_secondary_electric_mw,
            i_plasma_ignited=PlasmaIgnitionModel.NON_IGNITED,
        )


class HcdElectricTotalIgnited(HcdElectricTotal):
    """`i_plasma_ignited == 1` (`IGNITED`): zero, and two reads that are not reads.

    `current_drive.py:2294-2299` computes the sum and then overwrites it with `0.0`,
    under a comment that calls the reset a *"fudge"*. Declared as one node branching
    internally, the graph would carry
    `.heat_transport.p_hcd_primary_electric_mw` and `.p_hcd_secondary_electric_mw` as
    dependencies of a constant. Declared as an occupant, it reads nothing -- which is
    also the honest statement about `stellarator_helias.IN.DAT:126`, the reference run
    that sets this switch to `1`.
    """

    p_hcd_electric_total_mw = OutputInto(heat_transport)

    def __call__(self):
        return hcd_electric_total_mw(
            p_hcd_primary_electric_mw=0.0,
            p_hcd_secondary_electric_mw=0.0,
            i_plasma_ignited=PlasmaIgnitionModel.IGNITED,
        )
