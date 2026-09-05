"""Pure-functional port of `process/models/physics/current_drive.py`'s `CurrentDrive`.

Audit record: `functional_process/_audit/units/models/physics/current_drive.md`. Read it
first -- especially "A live PROCESS bug in two sibling arms" (`calculate_profile_y`
returns `None`) and "What this unit does *not* port and why", which between them say
exactly which of the eleven `i_hcd_primary` values this file can answer.

**Scope: the minimal closure that produces `_audit/tokamak_boundary.md`'s three
`.tokamak.current_drive` boundary reads** --
`.current_drive.p_hcd_ecrh_injected_total_mw`, `.current_drive.p_hcd_injected_total_mw`
and `.heat_transport.p_hcd_electric_total_mw` -- for the combinations the tracked
tokamak input files actually hold. `large_tokamak_eval.IN.DAT`'s: `i_hcd_primary = 10`
(`USER_INPUT_ELECTRON_CYCLOTRON`, the file's line 124), `i_hcd_secondary = 0`
(`NO_CURRENT_DRIVE`, PROCESS's default at `current_drive_variables.py:206` -- the file
never sets it), `i_hcd_calculations = 1` (default, `:223`) and `i_plasma_ignited = 0`
(`NON_IGNITED`, `physics_variables.py:881`). And, since 2026-08-27, the two spherical
tokamak files' (`spherical_tokamak_eval.IN.DAT:133`, `st_regression.IN.DAT:2522`):
`i_hcd_primary = 13` (`FREETHY_ELECTRON_CYCLOTRON`) with its nested `i_ecrh_wave_mode =
0` (O-mode -- both files set it explicitly, and it is also PROCESS's default at
`current_drive_variables.py:116`), same secondary/calculations/ignition values. The two
arms share every stage but the first: `CurrentDriveModel(13).method` is
`ELECTRON_CYCLOTRON`, so the wall-plug block is the one already ported. Every other
heating-and-current-drive scheme is UNPORTED; see the audit record for the per-value
reason.

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

from functional_process.models.stated import StatesValues
from functional_process.paths import current_drive, heat_transport, physics
from functional_process.physics.current_drive import (
    calculate_current_drive_ecrh_primary_no_secondary,
    calculate_current_drive_freethy_ecrh_primary_no_secondary,
    electron_cyclotron_primary_powers,
    freethy_electron_cyclotron_efficiency,
    fusion_gain,
    hcd_electric_total_mw,
    hcd_injected_power_total_mw,
    hcd_primary_injected_power_mw,
    hcd_secondary_driven_current,
    user_input_electron_cyclotron_efficiency,
)
from functional_process.vocabulary import PlasmaIgnitionModel

__all__ = [
    "calculate_current_drive_ecrh_primary_no_secondary",
    "calculate_current_drive_freethy_ecrh_primary_no_secondary",
]


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


class HcdPrimaryEfficiencyFreethyEcrhOMode(HcdPrimaryEfficiency):
    """`i_hcd_primary == 13` (`FREETHY_ELECTRON_CYCLOTRON`), O-mode.

    `spherical_tokamak_eval.IN.DAT:133` and `st_regression.IN.DAT:2522`'s value, the
    second occupant of this family. The nested `i_ecrh_wave_mode` is pinned to `0`
    (O-mode) -- both files set it explicitly (`:130` / `:2665`) and it is PROCESS's
    default (`current_drive_variables.py:116`). It is pinned rather than read because it
    is a switch, not a port (`_audit/naming_convention.md` § "switches are not ports"),
    and it stays a static kwarg of the shared pure function rather than splitting it
    because the two wave modes read identical variable sets -- see
    `freethy_electron_cyclotron_efficiency`'s docstring for the evidence. An X-mode
    occupant would be these same seven reads over the other branch; it is not written
    (`indat.py`'s `UNPORTED[("i_ecrh_wave_mode", 1)]`).

    Seven reads where model 10 has three -- including `feffcd`, the fudge factor model
    10's lambda conspicuously omits (`current_drive.py:1744-1747` has no `feffcd`;
    `:1759-1770` does).
    """

    eta_cd_hcd_primary = OutputInto(current_drive)

    def __call__(
        self,
        temp_plasma_electron_vol_avg_kev=From(physics),
        n_charge_plasma_effective_vol_avg=From(physics),
        rmajor=From(physics),
        nd_plasma_electrons_vol_avg=From(physics),
        b_plasma_toroidal_on_axis=From(physics),
        n_ecrh_harmonic=From(current_drive),
        feffcd=From(current_drive),
    ):
        return freethy_electron_cyclotron_efficiency(
            temp_plasma_electron_vol_avg_kev=temp_plasma_electron_vol_avg_kev,
            n_charge_plasma_effective_vol_avg=n_charge_plasma_effective_vol_avg,
            rmajor=rmajor,
            nd_plasma_electrons_vol_avg=nd_plasma_electrons_vol_avg,
            b_plasma_toroidal_on_axis=b_plasma_toroidal_on_axis,
            n_ecrh_harmonic=n_ecrh_harmonic,
            feffcd=feffcd,
            i_ecrh_wave_mode=0,
        )


class HcdSecondaryHeating(ExplicitFunction):
    """The family that owns what the *secondary* heating system contributes.

    `i_hcd_secondary` chooses among the same eleven efficiency models plus a twelfth
    value, `0`, meaning there is no secondary system at all.
    """


class HcdSecondaryHeatingNone(HcdSecondaryHeating, StatesValues):
    """`i_hcd_secondary == 0` (`NO_CURRENT_DRIVE`): the secondary contributes zero.

    PROCESS's default (`current_drive_variables.py:206`) and
    `large_tokamak_eval.IN.DAT`'s value, since that file never sets the switch.

    **A node that computes nothing, and that is the finding, not an accident.** (It
    reads its three outputs' statements and nothing else -- `models/stated.py` derives
    those, so the claim is unchanged from when it read literally nothing.) Of the three
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

    All three zeros are **stated** rather than literals in the body, so they reach the
    compiled program as arguments (`models/stated.py`, `_audit/optimise_design.md` §28,
    §34): each output is read at `^stated.<its place>` and supplied through the env.
    """

    eta_cd_hcd_secondary = OutputInto(current_drive)
    p_hcd_secondary_extra_heat_mw = OutputInto(current_drive)
    p_hcd_secondary_electric_mw = OutputInto(heat_transport)


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


class FusionGain(ExplicitFunction):
    """cottax node: `fusion_gain`, ports declared. Switch independent.

    Owns `.current_drive.big_q_plasma` on a **tokamak** graph. The stellarator's
    counterpart is `models/stellarator/heating.py::FusionGain`, registered in
    `models/stellarator/namespace.py`; the two never coexist, for the reason
    `TokamakCurrentDrive.electric_total`'s docstring gives about
    `.heat_transport.p_hcd_electric_total_mw` -- ownership is per-graph and the two
    device graphs are never assembled together.

    Nothing inside the model graph reads this path: its only reader is the problem
    layer -- `core/solver/objectives.py::objective_metric_5` (`i_figure_merit = -5`,
    `FUSION_GAIN_Q`, `st_regression.IN.DAT`'s) and
    `core/solver/constraints.py::constraint_28`. That is exactly why the node was
    missing for as long as it was: `boundary.py`'s pins are measured on the model graph,
    where an unread output is invisible.
    """

    big_q_plasma = OutputInto(current_drive)

    def __call__(
        self,
        p_fusion_total_mw=From(physics),
        p_hcd_injected_total_mw=From(current_drive),
        p_beam_orbit_loss_mw=From(current_drive),
        p_plasma_ohmic_mw=From(physics),
    ):
        return fusion_gain(
            p_fusion_total_mw=p_fusion_total_mw,
            p_hcd_injected_total_mw=p_hcd_injected_total_mw,
            p_beam_orbit_loss_mw=p_beam_orbit_loss_mw,
            p_plasma_ohmic_mw=p_plasma_ohmic_mw,
        )
