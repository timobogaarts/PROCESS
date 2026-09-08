"""Harness cases for `functional_process/cottax/physics/plasma_inductance.py`.

Audit record: `functional_process/_audit/units/models/physics/plasma_inductance.md`.

Four tier-1 contracts. Every one of them calls PROCESS's own function **directly**:
all four are `@staticmethod @nb.njit` on `PlasmaInductance` with explicit arguments and
no `self.data` access, so there is no adapter to write and no `DataStructure` back door
to close.

`PlasmaInductance.run()` itself has no contract. It is three assignments plus a dict
lookup on `i_ind_plasma_internal_norm` (`process/models/physics/physics.py`, lines
4712-4750), which is the switch dispatch the port replaces with occupant selection;
there is nothing left in it to compare once the four functions below agree.
"""

from functional_process.cottax._harness import Tier1Contract
from functional_process.cottax._harness.sample_store import FROM_FILE
from functional_process.cottax.physics.plasma_inductance import (
    calculate_internal_inductance_menard,
    calculate_internal_inductance_wesson,
    calculate_normalised_internal_inductance_iter_3,
    calculate_volt_second_requirements,
)
from process.models.physics.physics import PlasmaInductance


def _reference_volt_second_requirements(
    csawth,
    eps,
    f_c_plasma_inductive,
    ejima_coeff,
    kappa,
    rmajor,
    res_plasma,
    plasma_current,
    t_plant_pulse_fusion_ramp,
    t_plant_pulse_burn,
    ind_plasma_internal_norm,
):
    """`PlasmaInductance.calculate_volt_second_requirements`, called positionally.

    Positional because it is `@nb.njit`-compiled and PROCESS's own call site
    (`physics.py:938-950`) passes it that way; the port's keyword names are taken from
    the source signature, so the two orders are the same order.
    """
    return PlasmaInductance.calculate_volt_second_requirements(
        csawth,
        eps,
        f_c_plasma_inductive,
        ejima_coeff,
        kappa,
        rmajor,
        res_plasma,
        plasma_current,
        t_plant_pulse_fusion_ramp,
        t_plant_pulse_burn,
        ind_plasma_internal_norm,
    )


def _reference_internal_inductance_wesson(alphaj):
    """`PlasmaInductance.calculate_internal_inductance_wesson`."""
    return PlasmaInductance.calculate_internal_inductance_wesson(alphaj)


def _reference_internal_inductance_menard(kappa):
    """`PlasmaInductance.calculate_internal_inductance_menard`."""
    return PlasmaInductance.calculate_internal_inductance_menard(kappa)


def _reference_normalised_internal_inductance_iter_3(
    b_plasma_poloidal_vol_avg, c_plasma, vol_plasma, rmajor
):
    """`PlasmaInductance.calculate_normalised_internal_inductance_iter_3`."""
    return PlasmaInductance.calculate_normalised_internal_inductance_iter_3(
        b_plasma_poloidal_vol_avg, c_plasma, vol_plasma, rmajor
    )


class TestCalculateVoltSecondRequirements(Tier1Contract):
    """`calculate_volt_second_requirements` -> the `@nb.njit` staticmethod it ports.

    Produces `.physics.vs_plasma_ramp_required`, the boundary read that
    `models/pfcoil/currents.py::CSFluxSwing` declares.
    """

    audit_record = "models/physics/plasma_inductance.md"
    reference = _reference_volt_second_requirements
    ported = calculate_volt_second_requirements

    # Converged in-process `SingleRun` of `large_tokamak_eval.IN.DAT`. PROCESS reports
    # `vs_plasma_ramp_required = 279.0949824023401 Wb` at this point.
    samples = FROM_FILE

    fuzz = True


class TestCalculateInternalInductanceWesson(Tier1Contract):
    """`calculate_internal_inductance_wesson` -> its `@nb.njit` staticmethod.

    The live occupant on `large_tokamak_eval.IN.DAT`
    (`i_ind_plasma_internal_norm = 1`, set at line 311 of that file).
    """

    audit_record = "models/physics/plasma_inductance.md"
    reference = _reference_internal_inductance_wesson
    ported = calculate_internal_inductance_wesson

    samples = FROM_FILE

    # `1.65 + 0.89 * alphaj` must stay positive for the log; it does for any
    # `alphaj > -1.85`, and a current profile index is positive by construction.
    fuzz = True


class TestCalculateInternalInductanceMenard(Tier1Contract):
    """`calculate_internal_inductance_menard` -> its `@nb.njit` staticmethod.

    UNPORTED as an *occupant* (`i_ind_plasma_internal_norm = 2` is not this run's
    value), but the scaling itself is evaluated unconditionally by
    `PlasmaInductance.run()` and stored, so the function is ported and checked.
    """

    audit_record = "models/physics/plasma_inductance.md"
    reference = _reference_internal_inductance_menard
    ported = calculate_internal_inductance_menard

    samples = FROM_FILE

    fuzz = True


class TestCalculateNormalisedInternalInductanceIter3(Tier1Contract):
    """`calculate_normalised_internal_inductance_iter_3` -> its njit staticmethod."""

    audit_record = "models/physics/plasma_inductance.md"
    reference = _reference_normalised_internal_inductance_iter_3
    ported = calculate_normalised_internal_inductance_iter_3

    samples = FROM_FILE

    fuzz = True
