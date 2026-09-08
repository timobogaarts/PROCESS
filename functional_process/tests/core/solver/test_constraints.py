"""Harness cases for the ported constraint equations.

Covers every constraint ported in `constraints.py` (see that module's own docstring for
scope: everything PROCESS registers except 50/52, IFE-only).

`_reference_*` adapters bind a bare `DataStructure` with only the fields each
constraint's audited data footprint says it reads, then call PROCESS's own registered
constraint function through `ConstraintManager` -- the same closure that's actually
wired into the solver, not a re-implementation of it.
"""

import pytest

from functional_process.cottax._harness import (
    Tier1Contract,
    bounds_from_iteration_variables,
)
from functional_process.cottax._harness.process_reference import data_reference
from functional_process.cottax._harness.sample_store import FROM_FILE
from functional_process.cottax.core.solver.constraints import (
    constraint_1,
    constraint_2,
    constraint_3,
    constraint_4,
    constraint_5,
    constraint_6,
    constraint_7,
    constraint_8,
    constraint_9,
    constraint_11,
    constraint_12,
    constraint_13,
    constraint_14,
    constraint_15,
    constraint_16,
    constraint_17,
    constraint_18,
    constraint_19,
    constraint_20,
    constraint_21,
    constraint_22,
    constraint_23,
    constraint_24,
    constraint_25,
    constraint_26,
    constraint_27,
    constraint_28,
    constraint_29,
    constraint_30,
    constraint_31,
    constraint_32,
    constraint_33,
    constraint_34,
    constraint_35,
    constraint_36,
    constraint_37,
    constraint_39,
    constraint_40,
    constraint_41,
    constraint_42,
    constraint_43,
    constraint_44,
    constraint_45,
    constraint_46,
    constraint_48,
    constraint_51,
    constraint_53,
    constraint_54,
    constraint_56,
    constraint_59,
    constraint_60,
    constraint_61,
    constraint_62,
    constraint_63,
    constraint_64,
    constraint_65,
    constraint_66,
    constraint_67,
    constraint_68,
    constraint_72,
    constraint_73,
    constraint_74,
    constraint_75,
    constraint_76,
    constraint_77,
    constraint_78,
    constraint_79,
    constraint_80,
    constraint_81,
    constraint_82,
    constraint_83,
    constraint_84,
    constraint_85,
    constraint_86,
    constraint_87,
    constraint_88,
    constraint_89,
    constraint_90,
    constraint_91,
    constraint_92,
)
from process.core.exceptions import ProcessValueError
from process.core.model import DataStructure
from process.core.solver.constraints import ConstraintManager
from process.data_structure.physics_variables import PlasmaIgnitionModel
from process.models.physics.density_limit import DensityLimitModel
from process.models.physics.physics import BetaComponentLimits
from process.models.tfcoil.base import TFConductorModel


def _evaluate(constraint_id, data):
    """Call PROCESS's registered constraint function and flatten its result.

    Uses the actual `ConstraintRegistration` (populated by the `@ConstraintManager.
    register_constraint` decorator at import time), not a re-implementation, so this
    is exactly what the solver calls.
    """
    registration = ConstraintManager.get_constraint(constraint_id)
    result = registration.constraint_equation(registration, data)
    return (
        result.residual,
        result.normalised_residual,
        result.constraint_value,
        result.constraint_bound,
    )


_reference_constraint_1 = data_reference(lambda d: _evaluate(1, d))


class TestConstraint1(Tier1Contract):
    """`constraint_equation_1` -> `constraint_1`. `Compare`-shaped -- see the audit
    record's note that this is the first constraint of that shape ported.
    """

    audit_record = "core/solver/constraints.md"
    reference = _reference_constraint_1
    ported = constraint_1

    static_argnames = ()

    samples = FROM_FILE

    fuzz_bounds = {
        "beta_fast_alpha": (0.0, 0.01),
        "beta_beam": (0.0, 0.005),
        "nd_plasma_electrons_vol_avg": (1.0e19, 1.5e20),
        "temp_plasma_electron_density_weighted_kev": (2.0, 30.0),
        "nd_plasma_ions_total_vol_avg": (1.0e19, 1.5e20),
        "temp_plasma_ion_density_weighted_kev": (2.0, 30.0),
        "b_plasma_total": (1.0, 12.0),
        **bounds_from_iteration_variables("beta_total_vol_avg"),
    }


_reference_constraint_2 = data_reference(lambda d: _evaluate(2, d))


class TestConstraint2(Tier1Contract):
    """`constraint_equation_2` -> `constraint_2`."""

    audit_record = "core/solver/constraints.md"
    reference = _reference_constraint_2
    ported = constraint_2

    static_argnames = ("i_rad_loss", "i_plasma_ignited")

    _common = {
        "pden_electron_transport_loss_mw": 0.3,
        "pden_ion_transport_loss_mw": 0.25,
        "pden_plasma_rad_mw": 0.1,
        "pden_plasma_core_rad_mw": 0.05,
        "f_p_alpha_plasma_deposited": 0.95,
        "pden_alpha_total_mw": 0.4,
        "pden_non_alpha_charged_mw": 0.02,
        "pden_plasma_ohmic_mw": 0.01,
        "p_hcd_injected_total_mw": 50.0,
        "vol_plasma": 800.0,
    }

    samples = FROM_FILE

    fuzz_bounds = {
        "pden_electron_transport_loss_mw": (0.01, 2.0),
        "pden_ion_transport_loss_mw": (0.01, 2.0),
        "pden_plasma_rad_mw": (0.0, 1.0),
        "pden_plasma_core_rad_mw": (0.0, 1.0),
        "f_p_alpha_plasma_deposited": (0.5, 1.0),
        "pden_alpha_total_mw": (0.01, 2.0),
        "pden_non_alpha_charged_mw": (0.0, 0.5),
        "pden_plasma_ohmic_mw": (0.0, 0.5),
        "p_hcd_injected_total_mw": (0.0, 200.0),
        "vol_plasma": (100.0, 2000.0),
    }
    fuzz_fixed = {"i_rad_loss": 0, "i_plasma_ignited": 0}


_reference_constraint_3 = data_reference(lambda d: _evaluate(3, d))


class TestConstraint3(Tier1Contract):
    """`constraint_equation_3` -> `constraint_3`."""

    audit_record = "core/solver/constraints.md"
    reference = _reference_constraint_3
    ported = constraint_3

    static_argnames = ("i_plasma_ignited",)

    _common = {
        "pden_ion_transport_loss_mw": 0.25,
        "pden_ion_electron_equilibration_mw": 0.03,
        "f_p_alpha_plasma_deposited": 0.95,
        "f_pden_alpha_ions_mw": 0.15,
        "p_hcd_injected_ions_mw": 20.0,
        "vol_plasma": 800.0,
    }

    samples = FROM_FILE

    fuzz_bounds = {
        "pden_ion_transport_loss_mw": (0.01, 2.0),
        "pden_ion_electron_equilibration_mw": (-0.5, 0.5),
        "f_p_alpha_plasma_deposited": (0.5, 1.0),
        "f_pden_alpha_ions_mw": (0.01, 1.0),
        "p_hcd_injected_ions_mw": (0.0, 100.0),
        "vol_plasma": (100.0, 2000.0),
    }
    fuzz_fixed = {"i_plasma_ignited": 0}


_reference_constraint_4 = data_reference(lambda d: _evaluate(4, d))


class TestConstraint4(Tier1Contract):
    """`constraint_equation_4` -> `constraint_4`."""

    audit_record = "core/solver/constraints.md"
    reference = _reference_constraint_4
    ported = constraint_4

    static_argnames = ("i_rad_loss", "i_plasma_ignited")

    _common = {
        "pden_electron_transport_loss_mw": 0.3,
        "pden_plasma_rad_mw": 0.1,
        "pden_plasma_core_rad_mw": 0.05,
        "f_p_alpha_plasma_deposited": 0.95,
        "f_pden_alpha_electron_mw": 0.25,
        "pden_ion_electron_equilibration_mw": 0.03,
        "p_hcd_injected_electrons_mw": 30.0,
        "vol_plasma": 800.0,
    }

    samples = FROM_FILE

    fuzz_bounds = {
        "pden_electron_transport_loss_mw": (0.01, 2.0),
        "pden_plasma_rad_mw": (0.0, 1.0),
        "pden_plasma_core_rad_mw": (0.0, 1.0),
        "f_p_alpha_plasma_deposited": (0.5, 1.0),
        "f_pden_alpha_electron_mw": (0.01, 1.0),
        "pden_ion_electron_equilibration_mw": (-0.5, 0.5),
        "p_hcd_injected_electrons_mw": (0.0, 150.0),
        "vol_plasma": (100.0, 2000.0),
    }
    fuzz_fixed = {"i_rad_loss": 0, "i_plasma_ignited": 0}


_reference_constraint_5 = data_reference(lambda d: _evaluate(5, d))


class TestConstraint5(Tier1Contract):
    """`constraint_equation_5` -> `constraint_5`."""

    audit_record = "core/solver/constraints.md"
    reference = _reference_constraint_5
    ported = constraint_5

    static_argnames = ("i_density_limit",)

    _common = {
        "nd_plasma_electron_line": 8.0e19,
        "nd_plasma_electrons_vol_avg": 7.5e19,
        "nd_plasma_electrons_max": 9.0e19,
        "f_nd_plasma_electron_limit_max": 1.0,
    }

    samples = FROM_FILE

    fuzz_bounds = {
        "nd_plasma_electron_line": (1.0e19, 2.0e20),
        "nd_plasma_electrons_vol_avg": (1.0e19, 2.0e20),
        "nd_plasma_electrons_max": (1.0e19, 2.0e20),
        "f_nd_plasma_electron_limit_max": (0.5, 1.2),
    }
    fuzz_fixed = {"i_density_limit": int(DensityLimitModel.GREENWALD)}


_reference_constraint_6 = data_reference(lambda d: _evaluate(6, d))


class TestConstraint6(Tier1Contract):
    """`constraint_equation_6` -> `constraint_6`."""

    audit_record = "core/solver/constraints.md"
    reference = _reference_constraint_6
    ported = constraint_6

    static_argnames = ()

    samples = FROM_FILE

    fuzz = True


_reference_constraint_7 = data_reference(lambda d: _evaluate(7, d))


class TestConstraint7(Tier1Contract):
    """`constraint_equation_7` -> `constraint_7`.

    Only `NON_IGNITED` samples -- `i_plasma_ignited=IGNITED` raises in both PROCESS and
    the port (see `test_constraint_7_raises_when_ignited` below for that path, tested
    separately since `Tier1Contract`'s value-agreement samples aren't the right shape
    for a raise).
    """

    audit_record = "core/solver/constraints.md"
    reference = _reference_constraint_7
    ported = constraint_7

    static_argnames = ("i_plasma_ignited",)

    samples = FROM_FILE

    fuzz = True
    fuzz_fixed = {"i_plasma_ignited": int(PlasmaIgnitionModel.NON_IGNITED)}


def test_constraint_7_raises_when_ignited():
    """PROCESS's `constraint_equation_7` raises `ProcessValueError` if
    `i_plasma_ignited=IGNITED` (`constraints.py:517-518`); the port raises a plain
    `ValueError` for the same condition (see `constraint_7`'s own docstring). Confirms
    both actually raise, not just that the port's docstring claims they do.
    """
    data = DataStructure()
    data.physics.i_plasma_ignited = int(PlasmaIgnitionModel.IGNITED)
    data.physics.nd_beam_ions_out = 1.0e18
    data.physics.nd_beam_ions = 1.0e18
    with pytest.raises(ProcessValueError):
        _evaluate(7, data)

    with pytest.raises(ValueError, match="i_plasma_ignited"):
        constraint_7(int(PlasmaIgnitionModel.IGNITED), 1.0e18, 1.0e18)


_reference_constraint_8 = data_reference(lambda d: _evaluate(8, d))


class TestConstraint8(Tier1Contract):
    """`constraint_equation_8` -> `constraint_8`."""

    audit_record = "core/solver/constraints.md"
    reference = _reference_constraint_8
    ported = constraint_8

    static_argnames = ()

    samples = FROM_FILE

    fuzz = True


_reference_constraint_9 = data_reference(lambda d: _evaluate(9, d))


class TestConstraint9(Tier1Contract):
    """`constraint_equation_9` -> `constraint_9`."""

    audit_record = "core/solver/constraints.md"
    reference = _reference_constraint_9
    ported = constraint_9

    samples = FROM_FILE

    fuzz = True


_reference_constraint_11 = data_reference(lambda d: _evaluate(11, d))


class TestConstraint11(Tier1Contract):
    """`constraint_equation_11` -> `constraint_11`."""

    audit_record = "core/solver/constraints.md"
    reference = _reference_constraint_11
    ported = constraint_11

    samples = FROM_FILE

    fuzz = True


def _reference_constraint_12(vs_cs_pf_total_pulse, vs_plasma_total_required):
    """`vs_cs_pf_total_pulse` here is the port's already-sign-flipped argument --
    negate it back before writing to `data`, since PROCESS's own source stores the
    negative value and flips the sign at its own call site (see `batch1.md`).
    """
    data = DataStructure()
    data.pf_coil.vs_cs_pf_total_pulse = -vs_cs_pf_total_pulse
    data.physics.vs_plasma_total_required = vs_plasma_total_required
    return _evaluate(12, data)


class TestConstraint12(Tier1Contract):
    """`constraint_equation_12` -> `constraint_12`."""

    audit_record = "core/solver/constraints.md"
    reference = _reference_constraint_12
    ported = constraint_12

    samples = FROM_FILE

    fuzz = True


_reference_constraint_13 = data_reference(lambda d: _evaluate(13, d))


class TestConstraint13(Tier1Contract):
    """`constraint_equation_13` -> `constraint_13`."""

    audit_record = "core/solver/constraints.md"
    reference = _reference_constraint_13
    ported = constraint_13

    samples = FROM_FILE

    fuzz = True


_reference_constraint_14 = data_reference(lambda d: _evaluate(14, d))


class TestConstraint14(Tier1Contract):
    """`constraint_equation_14` -> `constraint_14`."""

    audit_record = "core/solver/constraints.md"
    reference = _reference_constraint_14
    ported = constraint_14

    samples = FROM_FILE

    fuzz = True


_reference_constraint_15 = data_reference(lambda d: _evaluate(15, d))


class TestConstraint15(Tier1Contract):
    """`constraint_equation_15` -> `constraint_15`."""

    audit_record = "core/solver/constraints.md"
    reference = _reference_constraint_15
    ported = constraint_15

    samples = FROM_FILE

    fuzz = True


_reference_constraint_16 = data_reference(lambda d: _evaluate(16, d))


class TestConstraint16(Tier1Contract):
    """`constraint_equation_16` -> `constraint_16`."""

    audit_record = "core/solver/constraints.md"
    reference = _reference_constraint_16
    ported = constraint_16

    samples = FROM_FILE

    fuzz = True


_reference_constraint_17 = data_reference(lambda d: _evaluate(17, d))


class TestConstraint17(Tier1Contract):
    """`constraint_equation_17` -> `constraint_17`."""

    audit_record = "core/solver/constraints.md"
    reference = _reference_constraint_17
    ported = constraint_17

    static_argnames = ("istell",)

    samples = FROM_FILE

    fuzz = True
    fuzz_fixed = {"istell": 1}


_reference_constraint_18 = data_reference(lambda d: _evaluate(18, d))


class TestConstraint18(Tier1Contract):
    """`constraint_equation_18` -> `constraint_18`."""

    audit_record = "core/solver/constraints.md"
    reference = _reference_constraint_18
    ported = constraint_18

    samples = FROM_FILE

    fuzz = True


_reference_constraint_19 = data_reference(lambda d: _evaluate(19, d))


class TestConstraint19(Tier1Contract):
    """`constraint_equation_19` -> `constraint_19`."""

    audit_record = "core/solver/constraints.md"
    reference = _reference_constraint_19
    ported = constraint_19

    samples = FROM_FILE

    fuzz = True


_reference_constraint_20 = data_reference(lambda d: _evaluate(20, d))


class TestConstraint20(Tier1Contract):
    """`constraint_equation_20` -> `constraint_20`."""

    audit_record = "core/solver/constraints.md"
    reference = _reference_constraint_20
    ported = constraint_20

    samples = FROM_FILE

    fuzz = True


_reference_constraint_21 = data_reference(lambda d: _evaluate(21, d))


class TestConstraint21(Tier1Contract):
    """`constraint_equation_21` -> `constraint_21`."""

    audit_record = "core/solver/constraints.md"
    reference = _reference_constraint_21
    ported = constraint_21

    samples = FROM_FILE

    fuzz = True


_reference_constraint_22 = data_reference(lambda d: _evaluate(22, d))


class TestConstraint22(Tier1Contract):
    """`constraint_equation_22` -> `constraint_22`."""

    audit_record = "core/solver/constraints.md"
    reference = _reference_constraint_22
    ported = constraint_22

    samples = FROM_FILE

    fuzz = True


_reference_constraint_23 = data_reference(lambda d: _evaluate(23, d))


class TestConstraint23(Tier1Contract):
    """`constraint_equation_23` -> `constraint_23`."""

    audit_record = "core/solver/constraints.md"
    reference = _reference_constraint_23
    ported = constraint_23

    samples = FROM_FILE

    fuzz = True


_reference_constraint_24 = data_reference(lambda d: _evaluate(24, d))


class TestConstraint24(Tier1Contract):
    """`constraint_equation_24` -> `constraint_24`."""

    audit_record = "core/solver/constraints.md"
    reference = _reference_constraint_24
    ported = constraint_24

    static_argnames = ("i_beta_component", "istell")

    _common = {
        "beta_total_vol_avg": 0.03,
        "beta_thermal_vol_avg": 0.025,
        "beta_beam": 0.002,
        "beta_toroidal_vol_avg": 0.028,
        "beta_vol_avg_max": 0.05,
    }

    samples = FROM_FILE

    fuzz_bounds = {
        **bounds_from_iteration_variables("beta_total_vol_avg"),
        "beta_thermal_vol_avg": (0.001, 0.05),
        "beta_beam": (0.0, 0.01),
        "beta_toroidal_vol_avg": (0.001, 0.05),
        "beta_vol_avg_max": (0.01, 0.1),
    }
    fuzz_fixed = {"i_beta_component": int(BetaComponentLimits.TOTAL), "istell": 0}


_reference_constraint_25 = data_reference(lambda d: _evaluate(25, d))


class TestConstraint25(Tier1Contract):
    """`constraint_equation_25` -> `constraint_25`."""

    audit_record = "core/solver/constraints.md"
    reference = _reference_constraint_25
    ported = constraint_25

    samples = FROM_FILE

    fuzz = True


_reference_constraint_26 = data_reference(lambda d: _evaluate(26, d))


class TestConstraint26(Tier1Contract):
    """`constraint_equation_26` -> `constraint_26`."""

    audit_record = "core/solver/constraints.md"
    reference = _reference_constraint_26
    ported = constraint_26

    samples = FROM_FILE

    fuzz = True


_reference_constraint_27 = data_reference(lambda d: _evaluate(27, d))


class TestConstraint27(Tier1Contract):
    """`constraint_equation_27` -> `constraint_27`."""

    audit_record = "core/solver/constraints.md"
    reference = _reference_constraint_27
    ported = constraint_27

    samples = FROM_FILE

    fuzz = True


_reference_constraint_28 = data_reference(lambda d: _evaluate(28, d))


class TestConstraint28(Tier1Contract):
    """`constraint_equation_28` -> `constraint_28`."""

    audit_record = "core/solver/constraints.md"
    reference = _reference_constraint_28
    ported = constraint_28

    static_argnames = ("i_plasma_ignited",)

    samples = FROM_FILE

    fuzz = True
    fuzz_fixed = {"i_plasma_ignited": int(PlasmaIgnitionModel.NON_IGNITED)}


def test_constraint_28_ported_raises_when_ignited():
    """Constraint 28 is not valid for an ignited plasma -- the port must raise."""
    with pytest.raises(ValueError, match="i_plasma_ignited"):
        constraint_28(int(PlasmaIgnitionModel.IGNITED), 15.0, 10.0)


def test_constraint_28_reference_raises_when_ignited():
    """Same precondition, checked against PROCESS's own registered function."""
    with pytest.raises(ProcessValueError):
        _reference_constraint_28(
            i_plasma_ignited=int(PlasmaIgnitionModel.IGNITED),
            big_q_plasma=15.0,
            big_q_plasma_min=10.0,
        )


_reference_constraint_29 = data_reference(lambda d: _evaluate(29, d))


class TestConstraint29(Tier1Contract):
    """`constraint_equation_29` -> `constraint_29`."""

    audit_record = "core/solver/constraints.md"
    reference = _reference_constraint_29
    ported = constraint_29

    samples = FROM_FILE

    fuzz = True


_reference_constraint_30 = data_reference(lambda d: _evaluate(30, d))


class TestConstraint30(Tier1Contract):
    """`constraint_equation_30` -> `constraint_30`."""

    audit_record = "core/solver/constraints.md"
    reference = _reference_constraint_30
    ported = constraint_30

    samples = FROM_FILE

    fuzz = True


_reference_constraint_31 = data_reference(lambda d: _evaluate(31, d))


class TestConstraint31(Tier1Contract):
    """`constraint_equation_31` -> `constraint_31`."""

    audit_record = "core/solver/constraints.md"
    reference = _reference_constraint_31
    ported = constraint_31

    samples = FROM_FILE

    fuzz = True


_reference_constraint_32 = data_reference(lambda d: _evaluate(32, d))


class TestConstraint32(Tier1Contract):
    """`constraint_equation_32` -> `constraint_32`."""

    audit_record = "core/solver/constraints.md"
    reference = _reference_constraint_32
    ported = constraint_32

    samples = FROM_FILE

    fuzz = True


_reference_constraint_33 = data_reference(lambda d: _evaluate(33, d))


class TestConstraint33(Tier1Contract):
    """`constraint_equation_33` -> `constraint_33`."""

    audit_record = "core/solver/constraints.md"
    reference = _reference_constraint_33
    ported = constraint_33

    samples = FROM_FILE

    fuzz = True


_reference_constraint_34 = data_reference(lambda d: _evaluate(34, d))


class TestConstraint34(Tier1Contract):
    """`constraint_equation_34` -> `constraint_34`."""

    audit_record = "core/solver/constraints.md"
    reference = _reference_constraint_34
    ported = constraint_34

    samples = FROM_FILE

    fuzz = True


_reference_constraint_35 = data_reference(lambda d: _evaluate(35, d))


class TestConstraint35(Tier1Contract):
    """`constraint_equation_35` -> `constraint_35`."""

    audit_record = "core/solver/constraints.md"
    reference = _reference_constraint_35
    ported = constraint_35

    samples = FROM_FILE

    fuzz = True


_reference_constraint_36 = data_reference(lambda d: _evaluate(36, d))


class TestConstraint36(Tier1Contract):
    """`constraint_equation_36` -> `constraint_36`."""

    audit_record = "core/solver/constraints.md"
    reference = _reference_constraint_36
    ported = constraint_36

    samples = FROM_FILE

    fuzz = True


_reference_constraint_37 = data_reference(lambda d: _evaluate(37, d))


class TestConstraint37(Tier1Contract):
    """`constraint_equation_37` -> `constraint_37`."""

    audit_record = "core/solver/constraints.md"
    reference = _reference_constraint_37
    ported = constraint_37

    samples = FROM_FILE

    fuzz = True


_reference_constraint_39 = data_reference(lambda d: _evaluate(39, d))


class TestConstraint39(Tier1Contract):
    """`constraint_equation_39` -> `constraint_39`."""

    audit_record = "core/solver/constraints.md"
    reference = _reference_constraint_39
    ported = constraint_39

    samples = FROM_FILE

    fuzz = True


_reference_constraint_40 = data_reference(lambda d: _evaluate(40, d))


class TestConstraint40(Tier1Contract):
    """`constraint_equation_40` -> `constraint_40`."""

    audit_record = "core/solver/constraints.md"
    reference = _reference_constraint_40
    ported = constraint_40

    samples = FROM_FILE

    fuzz = True


_reference_constraint_41 = data_reference(lambda d: _evaluate(41, d))


class TestConstraint41(Tier1Contract):
    """`constraint_equation_41` -> `constraint_41`."""

    audit_record = "core/solver/constraints.md"
    reference = _reference_constraint_41
    ported = constraint_41

    samples = FROM_FILE

    fuzz_bounds = {
        **bounds_from_iteration_variables("t_plant_pulse_plasma_current_ramp_up"),
        "t_current_ramp_up_min": (1.0, 500.0),
    }


_reference_constraint_42 = data_reference(lambda d: _evaluate(42, d))


class TestConstraint42(Tier1Contract):
    """`constraint_equation_42` -> `constraint_42`."""

    audit_record = "core/solver/constraints.md"
    reference = _reference_constraint_42
    ported = constraint_42

    samples = FROM_FILE

    fuzz = True


def _reference_constraint_43(i_tf_sup, temp_cp_average, tcpav2):
    data = DataStructure()
    data.physics.itart = 1  # required by PROCESS's own misuse guard, not a port param
    data.tfcoil.i_tf_sup = i_tf_sup
    data.tfcoil.temp_cp_average = temp_cp_average
    data.tfcoil.tcpav2 = tcpav2
    return _evaluate(43, data)


class TestConstraint43(Tier1Contract):
    """`constraint_equation_43` -> `constraint_43`."""

    audit_record = "core/solver/constraints.md"
    reference = _reference_constraint_43
    ported = constraint_43

    static_argnames = ("i_tf_sup",)

    samples = FROM_FILE

    fuzz = True
    fuzz_fixed = {"i_tf_sup": int(TFConductorModel.SUPERCONDUCTING)}


def _reference_constraint_44(i_tf_sup, temp_cp_max, temp_cp_peak):
    data = DataStructure()
    data.physics.itart = 1  # required by PROCESS's own misuse guard, not a port param
    data.tfcoil.i_tf_sup = i_tf_sup
    data.tfcoil.temp_cp_max = temp_cp_max
    data.tfcoil.temp_cp_peak = temp_cp_peak
    return _evaluate(44, data)


class TestConstraint44(Tier1Contract):
    """`constraint_equation_44` -> `constraint_44`."""

    audit_record = "core/solver/constraints.md"
    reference = _reference_constraint_44
    ported = constraint_44

    static_argnames = ("i_tf_sup",)

    samples = FROM_FILE

    fuzz = True
    fuzz_fixed = {"i_tf_sup": int(TFConductorModel.SUPERCONDUCTING)}


_reference_constraint_45 = data_reference(lambda d: _evaluate(45, d))


class TestConstraint45(Tier1Contract):
    """`constraint_manager_45` -> `constraint_45`."""

    audit_record = "core/solver/constraints.md"
    reference = _reference_constraint_45
    ported = constraint_45

    static_argnames = ("itart",)

    samples = FROM_FILE

    fuzz = True
    fuzz_fixed = {"itart": 1}


_reference_constraint_46 = data_reference(lambda d: _evaluate(46, d))


class TestConstraint46(Tier1Contract):
    """`constraint_equation_46` -> `constraint_46`."""

    audit_record = "core/solver/constraints.md"
    reference = _reference_constraint_46
    ported = constraint_46

    static_argnames = ("itart",)

    samples = FROM_FILE

    fuzz = True
    fuzz_fixed = {"itart": 1}


_reference_constraint_48 = data_reference(lambda d: _evaluate(48, d))


class TestConstraint48(Tier1Contract):
    """`constraint_equation_48` -> `constraint_48`."""

    audit_record = "core/solver/constraints.md"
    reference = _reference_constraint_48
    ported = constraint_48

    samples = FROM_FILE

    fuzz = True


_reference_constraint_51 = data_reference(lambda d: _evaluate(51, d))


class TestConstraint51(Tier1Contract):
    """`constraint_equation_51` -> `constraint_51`."""

    audit_record = "core/solver/constraints.md"
    reference = _reference_constraint_51
    ported = constraint_51

    samples = FROM_FILE

    fuzz = True


_reference_constraint_53 = data_reference(lambda d: _evaluate(53, d))


class TestConstraint53(Tier1Contract):
    """`constraint_equation_53` -> `constraint_53`."""

    audit_record = "core/solver/constraints.md"
    reference = _reference_constraint_53
    ported = constraint_53

    samples = FROM_FILE

    fuzz = True


_reference_constraint_54 = data_reference(lambda d: _evaluate(54, d))


class TestConstraint54(Tier1Contract):
    """`constraint_equation_54` -> `constraint_54`."""

    audit_record = "core/solver/constraints.md"
    reference = _reference_constraint_54
    ported = constraint_54

    samples = FROM_FILE

    fuzz = True


_reference_constraint_56 = data_reference(lambda d: _evaluate(56, d))


class TestConstraint56(Tier1Contract):
    """`constraint_equation_56` -> `constraint_56`."""

    audit_record = "core/solver/constraints.md"
    reference = _reference_constraint_56
    ported = constraint_56

    samples = FROM_FILE

    fuzz = True


_reference_constraint_59 = data_reference(lambda d: _evaluate(59, d))


class TestConstraint59(Tier1Contract):
    """`constraint_equation_59` -> `constraint_59`."""

    audit_record = "core/solver/constraints.md"
    reference = _reference_constraint_59
    ported = constraint_59

    samples = FROM_FILE

    fuzz = True


_reference_constraint_60 = data_reference(lambda d: _evaluate(60, d))


class TestConstraint60(Tier1Contract):
    """`constraint_equation_60` -> `constraint_60`."""

    audit_record = "core/solver/constraints.md"
    reference = _reference_constraint_60
    ported = constraint_60

    samples = FROM_FILE

    fuzz = True


_reference_constraint_61 = data_reference(lambda d: _evaluate(61, d))


class TestConstraint61(Tier1Contract):
    """`constraint_equation_61` -> `constraint_61`."""

    audit_record = "core/solver/constraints.md"
    reference = _reference_constraint_61
    ported = constraint_61

    samples = FROM_FILE

    fuzz = True


_reference_constraint_62 = data_reference(lambda d: _evaluate(62, d))


class TestConstraint62(Tier1Contract):
    """`constraint_equation_62` -> `constraint_62`."""

    audit_record = "core/solver/constraints.md"
    reference = _reference_constraint_62
    ported = constraint_62

    samples = FROM_FILE

    fuzz = True


_reference_constraint_63 = data_reference(lambda d: _evaluate(63, d))


class TestConstraint63(Tier1Contract):
    """`constraint_equation_63` -> `constraint_63`."""

    audit_record = "core/solver/constraints.md"
    reference = _reference_constraint_63
    ported = constraint_63

    samples = FROM_FILE

    fuzz = True


_reference_constraint_64 = data_reference(lambda d: _evaluate(64, d))


class TestConstraint64(Tier1Contract):
    """`constraint_equation_64` -> `constraint_64`."""

    audit_record = "core/solver/constraints.md"
    reference = _reference_constraint_64
    ported = constraint_64

    samples = FROM_FILE

    fuzz = True


_reference_constraint_65 = data_reference(lambda d: _evaluate(65, d))


class TestConstraint65(Tier1Contract):
    """`constraint_equation_65` -> `constraint_65`."""

    audit_record = "core/solver/constraints.md"
    reference = _reference_constraint_65
    ported = constraint_65

    samples = FROM_FILE

    fuzz = True


_reference_constraint_66 = data_reference(lambda d: _evaluate(66, d))


class TestConstraint66(Tier1Contract):
    """`constraint_equation_66` -> `constraint_66`."""

    audit_record = "core/solver/constraints.md"
    reference = _reference_constraint_66
    ported = constraint_66
    samples = FROM_FILE


_reference_constraint_67 = data_reference(lambda d: _evaluate(67, d))


class TestConstraint67(Tier1Contract):
    """`constraint_equation_67` -> `constraint_67`."""

    audit_record = "core/solver/constraints.md"
    reference = _reference_constraint_67
    ported = constraint_67
    samples = FROM_FILE


_reference_constraint_68 = data_reference(lambda d: _evaluate(68, d))


class TestConstraint68(Tier1Contract):
    """`constraint_equation_68` -> `constraint_68`."""

    audit_record = "core/solver/constraints.md"
    reference = _reference_constraint_68
    ported = constraint_68
    static_argnames = ("i_q95_fixed",)
    samples = FROM_FILE


_reference_constraint_72 = data_reference(lambda d: _evaluate(72, d))


class TestConstraint72(Tier1Contract):
    """`constraint_equation_72` -> `constraint_72`."""

    audit_record = "core/solver/constraints.md"
    reference = _reference_constraint_72
    ported = constraint_72
    static_argnames = ("i_tf_bucking", "i_tf_inside_cs")
    samples = FROM_FILE


_reference_constraint_73 = data_reference(lambda d: _evaluate(73, d))


class TestConstraint73(Tier1Contract):
    """`constraint_equation_73` -> `constraint_73`."""

    audit_record = "core/solver/constraints.md"
    reference = _reference_constraint_73
    ported = constraint_73
    samples = FROM_FILE


_reference_constraint_74 = data_reference(lambda d: _evaluate(74, d))


class TestConstraint74(Tier1Contract):
    """`constraint_equation_74` -> `constraint_74`."""

    audit_record = "core/solver/constraints.md"
    reference = _reference_constraint_74
    ported = constraint_74
    samples = FROM_FILE


_reference_constraint_75 = data_reference(lambda d: _evaluate(75, d))


class TestConstraint75(Tier1Contract):
    """`constraint_equation_75` -> `constraint_75`."""

    audit_record = "core/solver/constraints.md"
    reference = _reference_constraint_75
    ported = constraint_75
    samples = FROM_FILE


def _reference_constraint_76(
    kappa,
    triang,
    aspect,
    p_plasma_separatrix_mw,
    nd_plasma_electron_max_array_7,
    nd_plasma_separatrix_electron,
):
    data = DataStructure()
    data.physics.kappa = kappa
    data.physics.triang = triang
    data.physics.aspect = aspect
    data.physics.p_plasma_separatrix_mw = p_plasma_separatrix_mw
    # index 6 (0-indexed) is the 7th element -- see the ported function's own docstring.
    arr = list(data.physics.nd_plasma_electron_max_array)
    arr[6] = nd_plasma_electron_max_array_7
    data.physics.nd_plasma_electron_max_array = arr
    data.physics.nd_plasma_separatrix_electron = nd_plasma_separatrix_electron
    return _evaluate(76, data)


class TestConstraint76(Tier1Contract):
    """`constraint_equation_76` -> `constraint_76`."""

    audit_record = "core/solver/constraints.md"
    reference = _reference_constraint_76
    ported = constraint_76
    samples = FROM_FILE


_reference_constraint_77 = data_reference(lambda d: _evaluate(77, d))


class TestConstraint77(Tier1Contract):
    """`constraint_equation_77` -> `constraint_77`."""

    audit_record = "core/solver/constraints.md"
    reference = _reference_constraint_77
    ported = constraint_77

    samples = FROM_FILE

    fuzz = True


_reference_constraint_78 = data_reference(lambda d: _evaluate(78, d))


class TestConstraint78(Tier1Contract):
    """`constraint_equation_78` -> `constraint_78`."""

    audit_record = "core/solver/constraints.md"
    reference = _reference_constraint_78
    ported = constraint_78

    samples = FROM_FILE

    fuzz = True


_reference_constraint_79 = data_reference(lambda d: _evaluate(79, d))


class TestConstraint79(Tier1Contract):
    """`constraint_equation_79` -> `constraint_79`."""

    audit_record = "core/solver/constraints.md"
    reference = _reference_constraint_79
    ported = constraint_79

    samples = FROM_FILE

    fuzz = True


_reference_constraint_80 = data_reference(lambda d: _evaluate(80, d))


class TestConstraint80(Tier1Contract):
    """`constraint_equation_80` -> `constraint_80`."""

    audit_record = "core/solver/constraints.md"
    reference = _reference_constraint_80
    ported = constraint_80

    samples = FROM_FILE

    fuzz = True


_reference_constraint_81 = data_reference(lambda d: _evaluate(81, d))


class TestConstraint81(Tier1Contract):
    """`constraint_equation_81` -> `constraint_81`."""

    audit_record = "core/solver/constraints.md"
    reference = _reference_constraint_81
    ported = constraint_81

    samples = FROM_FILE

    fuzz = True


_reference_constraint_82 = data_reference(lambda d: _evaluate(82, d))


class TestConstraint82(Tier1Contract):
    """`constraint_equation_82` -> `constraint_82`."""

    audit_record = "core/solver/constraints.md"
    reference = _reference_constraint_82
    ported = constraint_82

    samples = FROM_FILE

    fuzz = True


_reference_constraint_83 = data_reference(lambda d: _evaluate(83, d))


class TestConstraint83(Tier1Contract):
    """`constraint_equation_83` -> `constraint_83`."""

    audit_record = "core/solver/constraints.md"
    reference = _reference_constraint_83
    ported = constraint_83

    samples = FROM_FILE

    fuzz = True


_reference_constraint_84 = data_reference(lambda d: _evaluate(84, d))


class TestConstraint84(Tier1Contract):
    """`constraint_equation_84` -> `constraint_84`."""

    audit_record = "core/solver/constraints.md"
    reference = _reference_constraint_84
    ported = constraint_84

    samples = FROM_FILE

    fuzz = True


_reference_constraint_85 = data_reference(lambda d: _evaluate(85, d))


class TestConstraint85(Tier1Contract):
    """`constraint_equation_85` -> `constraint_85`."""

    audit_record = "core/solver/constraints.md"
    reference = _reference_constraint_85
    ported = constraint_85

    static_argnames = ("i_cp_lifetime",)

    _common = {
        "cplife": 20.0,
        "cplife_input": 2.0,
        "life_div_fpy": 20.0,
        "life_blkt_fpy": 25.0,
        "life_plant": 30.0,
    }

    samples = FROM_FILE

    fuzz_bounds = {
        "cplife": (1.0, 60.0),
        "cplife_input": (1.0, 60.0),
        "life_div_fpy": (1.0, 60.0),
        "life_blkt_fpy": (1.0, 60.0),
        "life_plant": (1.0, 60.0),
    }
    fuzz_fixed = {"i_cp_lifetime": 1}


_reference_constraint_86 = data_reference(lambda d: _evaluate(86, d))


class TestConstraint86(Tier1Contract):
    """`constraint_equation_86` -> `constraint_86`."""

    audit_record = "core/solver/constraints.md"
    reference = _reference_constraint_86
    ported = constraint_86

    samples = FROM_FILE

    fuzz = True


_reference_constraint_87 = data_reference(lambda d: _evaluate(87, d))


class TestConstraint87(Tier1Contract):
    """`constraint_equation_87` -> `constraint_87`."""

    audit_record = "core/solver/constraints.md"
    reference = _reference_constraint_87
    ported = constraint_87

    samples = FROM_FILE
    fuzz = True


_reference_constraint_88 = data_reference(lambda d: _evaluate(88, d))


class TestConstraint88(Tier1Contract):
    """`constraint_equation_88` -> `constraint_88`."""

    audit_record = "core/solver/constraints.md"
    reference = _reference_constraint_88
    ported = constraint_88

    samples = FROM_FILE
    # `abs(str_wp)` is non-differentiable at str_wp == 0 -- keep fuzz bounds off zero,
    # same discipline as any other |.|-based constraint would need.
    fuzz = True


_reference_constraint_89 = data_reference(lambda d: _evaluate(89, d))


class TestConstraint89(Tier1Contract):
    """`constraint_equation_89` -> `constraint_89`."""

    audit_record = "core/solver/constraints.md"
    reference = _reference_constraint_89
    ported = constraint_89

    samples = FROM_FILE
    fuzz = True


_reference_constraint_90 = data_reference(lambda d: _evaluate(90, d))


class TestConstraint90(Tier1Contract):
    """`constraint_equation_90` -> `constraint_90`."""

    audit_record = "core/solver/constraints.md"
    reference = _reference_constraint_90
    ported = constraint_90

    static_argnames = ("ibkt_life", "bkt_life_csf")

    samples = FROM_FILE
    fuzz = True
    fuzz_fixed = {"ibkt_life": 0, "bkt_life_csf": 0.0}


_reference_constraint_91 = data_reference(lambda d: _evaluate(91, d))


class TestConstraint91(Tier1Contract):
    """`constraint_equation_91` -> `constraint_91`."""

    audit_record = "core/solver/constraints.md"
    reference = _reference_constraint_91
    ported = constraint_91

    static_argnames = ("i_plasma_ignited",)

    samples = FROM_FILE

    fuzz = True
    fuzz_fixed = {"i_plasma_ignited": int(PlasmaIgnitionModel.NON_IGNITED)}


_reference_constraint_92 = data_reference(lambda d: _evaluate(92, d))


class TestConstraint92(Tier1Contract):
    """`constraint_equation_92` -> `constraint_92`."""

    audit_record = "core/solver/constraints.md"
    reference = _reference_constraint_92
    ported = constraint_92

    samples = FROM_FILE
    fuzz = True
