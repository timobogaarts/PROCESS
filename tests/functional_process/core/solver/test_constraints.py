"""Harness cases for the ported constraint equations.

Covers every constraint ported in `constraints.py` (see that module's own docstring for
scope: everything PROCESS registers except 50/52, IFE-only).

`_reference_*` adapters bind a bare `DataStructure` with only the fields each
constraint's audited data footprint says it reads, then call PROCESS's own registered
constraint function through `ConstraintManager` -- the same closure that's actually
wired into the solver, not a re-implementation of it.
"""

import pytest

from functional_process._harness import (
    Tier1Contract,
    bounds_from_iteration_variables,
    legacy_sample,
)
from functional_process.core.solver.constraints import (
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
from process.data_structure.build_variables import TFCSRadialConfiguration
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


def _reference_constraint_1(
    beta_fast_alpha,
    beta_beam,
    nd_plasma_electrons_vol_avg,
    temp_plasma_electron_density_weighted_kev,
    nd_plasma_ions_total_vol_avg,
    temp_plasma_ion_density_weighted_kev,
    b_plasma_total,
    beta_total_vol_avg,
):
    data = DataStructure()
    data.physics.beta_fast_alpha = beta_fast_alpha
    data.physics.beta_beam = beta_beam
    data.physics.nd_plasma_electrons_vol_avg = nd_plasma_electrons_vol_avg
    data.physics.temp_plasma_electron_density_weighted_kev = (
        temp_plasma_electron_density_weighted_kev
    )
    data.physics.nd_plasma_ions_total_vol_avg = nd_plasma_ions_total_vol_avg
    data.physics.temp_plasma_ion_density_weighted_kev = (
        temp_plasma_ion_density_weighted_kev
    )
    data.physics.b_plasma_total = b_plasma_total
    data.physics.beta_total_vol_avg = beta_total_vol_avg
    return _evaluate(1, data)


class TestConstraint1(Tier1Contract):
    """`constraint_equation_1` -> `constraint_1`. `Compare`-shaped -- see the audit
    record's note that this is the first constraint of that shape ported.
    """

    audit_record = "core/solver/constraints.md"
    reference = _reference_constraint_1
    ported = constraint_1

    static_argnames = ()

    samples = [
        legacy_sample(
            "balanced",
            beta_fast_alpha=0.002,
            beta_beam=0.0005,
            nd_plasma_electrons_vol_avg=7.5e19,
            temp_plasma_electron_density_weighted_kev=13.0,
            nd_plasma_ions_total_vol_avg=7.2e19,
            temp_plasma_ion_density_weighted_kev=13.5,
            b_plasma_total=5.0,
            beta_total_vol_avg=0.03,
        ),
        legacy_sample(
            "low-field",
            beta_fast_alpha=0.001,
            beta_beam=0.0,
            nd_plasma_electrons_vol_avg=5.0e19,
            temp_plasma_electron_density_weighted_kev=10.0,
            nd_plasma_ions_total_vol_avg=4.8e19,
            temp_plasma_ion_density_weighted_kev=10.5,
            b_plasma_total=3.0,
            beta_total_vol_avg=0.04,
        ),
    ]

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


def _reference_constraint_2(
    i_rad_loss,
    i_plasma_ignited,
    pden_electron_transport_loss_mw,
    pden_ion_transport_loss_mw,
    pden_plasma_rad_mw,
    pden_plasma_core_rad_mw,
    f_p_alpha_plasma_deposited,
    pden_alpha_total_mw,
    pden_non_alpha_charged_mw,
    pden_plasma_ohmic_mw,
    p_hcd_injected_total_mw,
    vol_plasma,
):
    data = DataStructure()
    data.physics.i_rad_loss = i_rad_loss
    data.physics.i_plasma_ignited = i_plasma_ignited
    data.physics.pden_electron_transport_loss_mw = pden_electron_transport_loss_mw
    data.physics.pden_ion_transport_loss_mw = pden_ion_transport_loss_mw
    data.physics.pden_plasma_rad_mw = pden_plasma_rad_mw
    data.physics.pden_plasma_core_rad_mw = pden_plasma_core_rad_mw
    data.physics.f_p_alpha_plasma_deposited = f_p_alpha_plasma_deposited
    data.physics.pden_alpha_total_mw = pden_alpha_total_mw
    data.physics.pden_non_alpha_charged_mw = pden_non_alpha_charged_mw
    data.physics.pden_plasma_ohmic_mw = pden_plasma_ohmic_mw
    data.current_drive.p_hcd_injected_total_mw = p_hcd_injected_total_mw
    data.physics.vol_plasma = vol_plasma
    return _evaluate(2, data)


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

    samples = [
        legacy_sample("non-ignited-rad0", i_rad_loss=0, i_plasma_ignited=0, **_common),
        legacy_sample("non-ignited-rad1", i_rad_loss=1, i_plasma_ignited=0, **_common),
        legacy_sample("non-ignited-rad2", i_rad_loss=2, i_plasma_ignited=0, **_common),
        legacy_sample("ignited-rad0", i_rad_loss=0, i_plasma_ignited=1, **_common),
    ]

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


def _reference_constraint_3(
    i_plasma_ignited,
    pden_ion_transport_loss_mw,
    pden_ion_electron_equilibration_mw,
    f_p_alpha_plasma_deposited,
    f_pden_alpha_ions_mw,
    p_hcd_injected_ions_mw,
    vol_plasma,
):
    data = DataStructure()
    data.physics.i_plasma_ignited = i_plasma_ignited
    data.physics.pden_ion_transport_loss_mw = pden_ion_transport_loss_mw
    data.physics.pden_ion_electron_equilibration_mw = pden_ion_electron_equilibration_mw
    data.physics.f_p_alpha_plasma_deposited = f_p_alpha_plasma_deposited
    data.physics.f_pden_alpha_ions_mw = f_pden_alpha_ions_mw
    data.current_drive.p_hcd_injected_ions_mw = p_hcd_injected_ions_mw
    data.physics.vol_plasma = vol_plasma
    return _evaluate(3, data)


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

    samples = [
        legacy_sample("non-ignited", i_plasma_ignited=0, **_common),
        legacy_sample("ignited", i_plasma_ignited=1, **_common),
    ]

    fuzz_bounds = {
        "pden_ion_transport_loss_mw": (0.01, 2.0),
        "pden_ion_electron_equilibration_mw": (-0.5, 0.5),
        "f_p_alpha_plasma_deposited": (0.5, 1.0),
        "f_pden_alpha_ions_mw": (0.01, 1.0),
        "p_hcd_injected_ions_mw": (0.0, 100.0),
        "vol_plasma": (100.0, 2000.0),
    }
    fuzz_fixed = {"i_plasma_ignited": 0}


def _reference_constraint_4(
    i_rad_loss,
    i_plasma_ignited,
    pden_electron_transport_loss_mw,
    pden_plasma_rad_mw,
    pden_plasma_core_rad_mw,
    f_p_alpha_plasma_deposited,
    f_pden_alpha_electron_mw,
    pden_ion_electron_equilibration_mw,
    p_hcd_injected_electrons_mw,
    vol_plasma,
):
    data = DataStructure()
    data.physics.i_rad_loss = i_rad_loss
    data.physics.i_plasma_ignited = i_plasma_ignited
    data.physics.pden_electron_transport_loss_mw = pden_electron_transport_loss_mw
    data.physics.pden_plasma_rad_mw = pden_plasma_rad_mw
    data.physics.pden_plasma_core_rad_mw = pden_plasma_core_rad_mw
    data.physics.f_p_alpha_plasma_deposited = f_p_alpha_plasma_deposited
    data.physics.f_pden_alpha_electron_mw = f_pden_alpha_electron_mw
    data.physics.pden_ion_electron_equilibration_mw = pden_ion_electron_equilibration_mw
    data.current_drive.p_hcd_injected_electrons_mw = p_hcd_injected_electrons_mw
    data.physics.vol_plasma = vol_plasma
    return _evaluate(4, data)


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

    samples = [
        legacy_sample("non-ignited-rad0", i_rad_loss=0, i_plasma_ignited=0, **_common),
        legacy_sample("non-ignited-rad1", i_rad_loss=1, i_plasma_ignited=0, **_common),
        legacy_sample("ignited-rad2", i_rad_loss=2, i_plasma_ignited=1, **_common),
    ]

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


def _reference_constraint_5(
    i_density_limit,
    nd_plasma_electron_line,
    nd_plasma_electrons_vol_avg,
    nd_plasma_electrons_max,
    f_nd_plasma_electron_limit_max,
):
    data = DataStructure()
    data.physics.i_density_limit = i_density_limit
    data.physics.nd_plasma_electron_line = nd_plasma_electron_line
    data.physics.nd_plasma_electrons_vol_avg = nd_plasma_electrons_vol_avg
    data.physics.nd_plasma_electrons_max = nd_plasma_electrons_max
    data.constraints.f_nd_plasma_electron_limit_max = f_nd_plasma_electron_limit_max
    return _evaluate(5, data)


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

    samples = [
        legacy_sample(
            "greenwald", i_density_limit=int(DensityLimitModel.GREENWALD), **_common
        ),
        legacy_sample("asdex", i_density_limit=int(DensityLimitModel.ASDEX), **_common),
    ]

    fuzz_bounds = {
        "nd_plasma_electron_line": (1.0e19, 2.0e20),
        "nd_plasma_electrons_vol_avg": (1.0e19, 2.0e20),
        "nd_plasma_electrons_max": (1.0e19, 2.0e20),
        "f_nd_plasma_electron_limit_max": (0.5, 1.2),
    }
    fuzz_fixed = {"i_density_limit": int(DensityLimitModel.GREENWALD)}


def _reference_constraint_6(beta_poloidal_eps, beta_poloidal_eps_max):
    data = DataStructure()
    data.physics.beta_poloidal_eps = beta_poloidal_eps
    data.physics.beta_poloidal_eps_max = beta_poloidal_eps_max
    return _evaluate(6, data)


class TestConstraint6(Tier1Contract):
    """`constraint_equation_6` -> `constraint_6`."""

    audit_record = "core/solver/constraints.md"
    reference = _reference_constraint_6
    ported = constraint_6

    static_argnames = ()

    samples = [
        legacy_sample("feasible", beta_poloidal_eps=0.5, beta_poloidal_eps_max=1.38),
        legacy_sample("infeasible", beta_poloidal_eps=1.5, beta_poloidal_eps_max=1.38),
    ]

    fuzz_bounds = {
        "beta_poloidal_eps": (0.01, 3.0),
        "beta_poloidal_eps_max": (0.5, 2.0),
    }


def _reference_constraint_7(i_plasma_ignited, nd_beam_ions_out, nd_beam_ions):
    data = DataStructure()
    data.physics.i_plasma_ignited = i_plasma_ignited
    data.physics.nd_beam_ions_out = nd_beam_ions_out
    data.physics.nd_beam_ions = nd_beam_ions
    return _evaluate(7, data)


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

    samples = [
        legacy_sample(
            "matched",
            i_plasma_ignited=int(PlasmaIgnitionModel.NON_IGNITED),
            nd_beam_ions_out=2.0e18,
            nd_beam_ions=2.0e18,
        ),
        legacy_sample(
            "mismatched",
            i_plasma_ignited=int(PlasmaIgnitionModel.NON_IGNITED),
            nd_beam_ions_out=1.5e18,
            nd_beam_ions=2.0e18,
        ),
    ]

    fuzz_bounds = {
        "nd_beam_ions_out": (0.0, 1.0e19),
        "nd_beam_ions": (0.0, 1.0e19),
    }
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


def _reference_constraint_8(pflux_fw_neutron_mw, pflux_fw_neutron_max_mw):
    data = DataStructure()
    data.physics.pflux_fw_neutron_mw = pflux_fw_neutron_mw
    data.constraints.pflux_fw_neutron_max_mw = pflux_fw_neutron_max_mw
    return _evaluate(8, data)


class TestConstraint8(Tier1Contract):
    """`constraint_equation_8` -> `constraint_8`."""

    audit_record = "core/solver/constraints.md"
    reference = _reference_constraint_8
    ported = constraint_8

    static_argnames = ()

    samples = [
        legacy_sample("feasible", pflux_fw_neutron_mw=0.8, pflux_fw_neutron_max_mw=1.0),
        legacy_sample(
            "infeasible", pflux_fw_neutron_mw=1.2, pflux_fw_neutron_max_mw=1.0
        ),
    ]

    fuzz_bounds = {
        "pflux_fw_neutron_mw": (0.0, 3.0),
        "pflux_fw_neutron_max_mw": (0.5, 2.0),
    }


def _reference_constraint_9(p_fusion_total_mw, p_fusion_total_max_mw):
    data = DataStructure()
    data.physics.p_fusion_total_mw = p_fusion_total_mw
    data.constraints.p_fusion_total_max_mw = p_fusion_total_max_mw
    return _evaluate(9, data)


class TestConstraint9(Tier1Contract):
    """`constraint_equation_9` -> `constraint_9`."""

    audit_record = "core/solver/constraints.md"
    reference = _reference_constraint_9
    ported = constraint_9

    samples = [
        legacy_sample(
            "feasible", p_fusion_total_mw=2000.0, p_fusion_total_max_mw=3000.0
        ),
        legacy_sample(
            "infeasible", p_fusion_total_mw=3500.0, p_fusion_total_max_mw=3000.0
        ),
    ]

    fuzz_bounds = {
        "p_fusion_total_mw": (1.0, 5000.0),
        "p_fusion_total_max_mw": (1.0, 5000.0),
    }


def _reference_constraint_11(rbld, rmajor):
    data = DataStructure()
    data.build.rbld = rbld
    data.physics.rmajor = rmajor
    return _evaluate(11, data)


class TestConstraint11(Tier1Contract):
    """`constraint_equation_11` -> `constraint_11`."""

    audit_record = "core/solver/constraints.md"
    reference = _reference_constraint_11
    ported = constraint_11

    samples = [
        legacy_sample("consistent", rbld=8.0, rmajor=8.0),
        legacy_sample("inconsistent", rbld=8.2, rmajor=8.0),
    ]

    fuzz_bounds = {"rbld": (1.0, 20.0), "rmajor": (1.0, 20.0)}


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

    samples = [
        legacy_sample(
            "feasible", vs_cs_pf_total_pulse=120.0, vs_plasma_total_required=100.0
        ),
        legacy_sample(
            "infeasible", vs_cs_pf_total_pulse=80.0, vs_plasma_total_required=100.0
        ),
    ]

    fuzz_bounds = {
        "vs_cs_pf_total_pulse": (1.0, 500.0),
        "vs_plasma_total_required": (1.0, 500.0),
    }


def _reference_constraint_13(t_plant_pulse_burn, t_burn_min):
    data = DataStructure()
    data.times.t_plant_pulse_burn = t_plant_pulse_burn
    data.constraints.t_burn_min = t_burn_min
    return _evaluate(13, data)


class TestConstraint13(Tier1Contract):
    """`constraint_equation_13` -> `constraint_13`."""

    audit_record = "core/solver/constraints.md"
    reference = _reference_constraint_13
    ported = constraint_13

    samples = [
        legacy_sample("feasible", t_plant_pulse_burn=3.15576e7, t_burn_min=1000.0),
        legacy_sample("infeasible", t_plant_pulse_burn=500.0, t_burn_min=1000.0),
    ]

    fuzz_bounds = {
        "t_plant_pulse_burn": (1.0, 4.0e7),
        "t_burn_min": (1.0, 1.0e4),
    }


def _reference_constraint_14(
    n_beam_decay_lengths_core, n_beam_decay_lengths_core_required
):
    data = DataStructure()
    data.current_drive.n_beam_decay_lengths_core = n_beam_decay_lengths_core
    data.current_drive.n_beam_decay_lengths_core_required = (
        n_beam_decay_lengths_core_required
    )
    return _evaluate(14, data)


class TestConstraint14(Tier1Contract):
    """`constraint_equation_14` -> `constraint_14`."""

    audit_record = "core/solver/constraints.md"
    reference = _reference_constraint_14
    ported = constraint_14

    samples = [
        legacy_sample(
            "consistent",
            n_beam_decay_lengths_core=1.5,
            n_beam_decay_lengths_core_required=1.5,
        ),
        legacy_sample(
            "inconsistent",
            n_beam_decay_lengths_core=1.7,
            n_beam_decay_lengths_core_required=1.5,
        ),
    ]

    fuzz_bounds = {
        "n_beam_decay_lengths_core": (0.1, 5.0),
        "n_beam_decay_lengths_core_required": (0.1, 5.0),
    }


def _reference_constraint_15(
    p_plasma_separatrix_mw, p_l_h_threshold_mw, f_h_mode_margin
):
    data = DataStructure()
    data.physics.p_plasma_separatrix_mw = p_plasma_separatrix_mw
    data.physics.p_l_h_threshold_mw = p_l_h_threshold_mw
    data.constraints.f_h_mode_margin = f_h_mode_margin
    return _evaluate(15, data)


class TestConstraint15(Tier1Contract):
    """`constraint_equation_15` -> `constraint_15`."""

    audit_record = "core/solver/constraints.md"
    reference = _reference_constraint_15
    ported = constraint_15

    samples = [
        legacy_sample(
            "feasible",
            p_plasma_separatrix_mw=50.0,
            p_l_h_threshold_mw=40.0,
            f_h_mode_margin=1.0,
        ),
        legacy_sample(
            "infeasible",
            p_plasma_separatrix_mw=30.0,
            p_l_h_threshold_mw=40.0,
            f_h_mode_margin=1.2,
        ),
    ]

    fuzz_bounds = {
        "p_plasma_separatrix_mw": (1.0, 200.0),
        "p_l_h_threshold_mw": (1.0, 200.0),
        "f_h_mode_margin": (0.5, 2.0),
    }


def _reference_constraint_16(p_plant_electric_net_mw, p_plant_electric_net_required_mw):
    data = DataStructure()
    data.heat_transport.p_plant_electric_net_mw = p_plant_electric_net_mw
    data.constraints.p_plant_electric_net_required_mw = p_plant_electric_net_required_mw
    return _evaluate(16, data)


class TestConstraint16(Tier1Contract):
    """`constraint_equation_16` -> `constraint_16`."""

    audit_record = "core/solver/constraints.md"
    reference = _reference_constraint_16
    ported = constraint_16

    samples = [
        legacy_sample(
            "feasible",
            p_plant_electric_net_mw=550.0,
            p_plant_electric_net_required_mw=500.0,
        ),
        legacy_sample(
            "infeasible",
            p_plant_electric_net_mw=450.0,
            p_plant_electric_net_required_mw=500.0,
        ),
    ]

    fuzz_bounds = {
        "p_plant_electric_net_mw": (1.0, 2000.0),
        "p_plant_electric_net_required_mw": (1.0, 2000.0),
    }


def _reference_constraint_17(
    istell,
    f_p_plasma_separatrix_rad,
    f_p_plasma_separatrix_rad_max,
    psolradmw,
    p_plasma_heating_total_mw,
):
    """Call PROCESS's `constraint_equation_17` through the port's signature."""
    data = DataStructure()
    data.stellarator.istell = istell
    data.physics.f_p_plasma_separatrix_rad = f_p_plasma_separatrix_rad
    data.constraints.f_p_plasma_separatrix_rad_max = f_p_plasma_separatrix_rad_max
    data.physics.psolradmw = psolradmw
    data.physics.p_plasma_heating_total_mw = p_plasma_heating_total_mw
    return _evaluate(17, data)


class TestConstraint17(Tier1Contract):
    """`constraint_equation_17` -> `constraint_17`."""

    audit_record = "core/solver/constraints.md"
    reference = _reference_constraint_17
    ported = constraint_17

    static_argnames = ("istell",)

    samples = [
        legacy_sample(
            "tokamak",
            istell=0,
            f_p_plasma_separatrix_rad=0.5,
            f_p_plasma_separatrix_rad_max=0.85,
            # Unused on this branch -- present because the signature is unconditional.
            psolradmw=0.0,
            p_plasma_heating_total_mw=300.0,
        ),
        legacy_sample(
            "stellarator-feasible",
            istell=1,
            f_p_plasma_separatrix_rad=0.6,
            f_p_plasma_separatrix_rad_max=0.85,
            psolradmw=30.0,
            p_plasma_heating_total_mw=300.0,
        ),
        legacy_sample(
            "stellarator-infeasible",
            istell=1,
            f_p_plasma_separatrix_rad=0.95,
            f_p_plasma_separatrix_rad_max=0.85,
            psolradmw=10.0,
            p_plasma_heating_total_mw=300.0,
        ),
    ]

    fuzz_bounds = {
        "f_p_plasma_separatrix_rad": (0.0, 1.0),
        "f_p_plasma_separatrix_rad_max": (0.1, 1.0),
        "psolradmw": (0.0, 200.0),
        "p_plasma_heating_total_mw": (10.0, 1000.0),
    }
    fuzz_fixed = {"istell": 1}


def _reference_constraint_18(pflux_div_heat_load_mw, pflux_div_heat_load_max_mw):
    data = DataStructure()
    data.divertor.pflux_div_heat_load_mw = pflux_div_heat_load_mw
    data.divertor.pflux_div_heat_load_max_mw = pflux_div_heat_load_max_mw
    return _evaluate(18, data)


class TestConstraint18(Tier1Contract):
    """`constraint_equation_18` -> `constraint_18`."""

    audit_record = "core/solver/constraints.md"
    reference = _reference_constraint_18
    ported = constraint_18

    samples = [
        legacy_sample(
            "feasible", pflux_div_heat_load_mw=8.0, pflux_div_heat_load_max_mw=10.0
        ),
        legacy_sample(
            "infeasible", pflux_div_heat_load_mw=12.0, pflux_div_heat_load_max_mw=10.0
        ),
    ]

    fuzz_bounds = {
        "pflux_div_heat_load_mw": (0.1, 20.0),
        "pflux_div_heat_load_max_mw": (0.1, 20.0),
    }


def _reference_constraint_19(p_cp_resistive_mw, p_tf_leg_resistive_mw, mvalim):
    data = DataStructure()
    data.tfcoil.p_cp_resistive_mw = p_cp_resistive_mw
    data.tfcoil.p_tf_leg_resistive_mw = p_tf_leg_resistive_mw
    data.constraints.mvalim = mvalim
    return _evaluate(19, data)


class TestConstraint19(Tier1Contract):
    """`constraint_equation_19` -> `constraint_19`."""

    audit_record = "core/solver/constraints.md"
    reference = _reference_constraint_19
    ported = constraint_19

    samples = [
        legacy_sample(
            "feasible", p_cp_resistive_mw=10.0, p_tf_leg_resistive_mw=5.0, mvalim=40.0
        ),
        legacy_sample(
            "infeasible", p_cp_resistive_mw=30.0, p_tf_leg_resistive_mw=20.0, mvalim=40.0
        ),
    ]

    fuzz_bounds = {
        "p_cp_resistive_mw": (0.0, 100.0),
        "p_tf_leg_resistive_mw": (0.0, 100.0),
        "mvalim": (1.0, 200.0),
    }


def _reference_constraint_20(radius_beam_tangency, radius_beam_tangency_max):
    data = DataStructure()
    data.current_drive.radius_beam_tangency = radius_beam_tangency
    data.current_drive.radius_beam_tangency_max = radius_beam_tangency_max
    return _evaluate(20, data)


class TestConstraint20(Tier1Contract):
    """`constraint_equation_20` -> `constraint_20`."""

    audit_record = "core/solver/constraints.md"
    reference = _reference_constraint_20
    ported = constraint_20

    samples = [
        legacy_sample(
            "feasible", radius_beam_tangency=6.0, radius_beam_tangency_max=8.0
        ),
        legacy_sample(
            "infeasible", radius_beam_tangency=9.0, radius_beam_tangency_max=8.0
        ),
    ]

    fuzz_bounds = {
        "radius_beam_tangency": (0.1, 20.0),
        "radius_beam_tangency_max": (0.1, 20.0),
    }


def _reference_constraint_21(rminor, rminor_min):
    data = DataStructure()
    data.physics.rminor = rminor
    data.build.rminor_min = rminor_min
    return _evaluate(21, data)


class TestConstraint21(Tier1Contract):
    """`constraint_equation_21` -> `constraint_21`."""

    audit_record = "core/solver/constraints.md"
    reference = _reference_constraint_21
    ported = constraint_21

    samples = [
        legacy_sample("feasible", rminor=1.8, rminor_min=0.25),
        legacy_sample("infeasible", rminor=0.1, rminor_min=0.25),
    ]

    fuzz_bounds = {"rminor": (0.01, 5.0), "rminor_min": (0.01, 2.0)}


def _reference_constraint_22(
    p_l_h_threshold_mw, f_l_mode_margin, p_plasma_separatrix_mw
):
    data = DataStructure()
    data.physics.p_l_h_threshold_mw = p_l_h_threshold_mw
    data.constraints.f_l_mode_margin = f_l_mode_margin
    data.physics.p_plasma_separatrix_mw = p_plasma_separatrix_mw
    return _evaluate(22, data)


class TestConstraint22(Tier1Contract):
    """`constraint_equation_22` -> `constraint_22`."""

    audit_record = "core/solver/constraints.md"
    reference = _reference_constraint_22
    ported = constraint_22

    samples = [
        legacy_sample(
            "feasible-no-margin",
            p_l_h_threshold_mw=50.0,
            f_l_mode_margin=1.0,
            p_plasma_separatrix_mw=40.0,
        ),
        legacy_sample(
            "infeasible-with-margin",
            p_l_h_threshold_mw=50.0,
            f_l_mode_margin=1.2,
            p_plasma_separatrix_mw=45.0,
        ),
    ]

    fuzz_bounds = {
        "p_l_h_threshold_mw": (0.1, 500.0),
        "f_l_mode_margin": (0.5, 2.0),
        "p_plasma_separatrix_mw": (0.1, 500.0),
    }


def _reference_constraint_23(
    rminor,
    dr_fw_plasma_gap_outboard,
    dr_fw_outboard,
    dr_blkt_outboard,
    f_r_conducting_wall,
):
    data = DataStructure()
    data.physics.rminor = rminor
    data.build.dr_fw_plasma_gap_outboard = dr_fw_plasma_gap_outboard
    data.build.dr_fw_outboard = dr_fw_outboard
    data.build.dr_blkt_outboard = dr_blkt_outboard
    data.physics.f_r_conducting_wall = f_r_conducting_wall
    return _evaluate(23, data)


class TestConstraint23(Tier1Contract):
    """`constraint_equation_23` -> `constraint_23`."""

    audit_record = "core/solver/constraints.md"
    reference = _reference_constraint_23
    ported = constraint_23

    samples = [
        legacy_sample(
            "feasible",
            rminor=1.8,
            dr_fw_plasma_gap_outboard=0.05,
            dr_fw_outboard=0.02,
            dr_blkt_outboard=0.4,
            f_r_conducting_wall=1.35,
        ),
        legacy_sample(
            "infeasible",
            rminor=1.0,
            dr_fw_plasma_gap_outboard=0.5,
            dr_fw_outboard=0.5,
            dr_blkt_outboard=0.9,
            f_r_conducting_wall=1.05,
        ),
    ]

    fuzz_bounds = {
        "rminor": (0.1, 5.0),
        "dr_fw_plasma_gap_outboard": (0.0, 1.0),
        "dr_fw_outboard": (0.0, 1.0),
        "dr_blkt_outboard": (0.0, 2.0),
        "f_r_conducting_wall": (1.0, 2.0),
    }


def _reference_constraint_24(
    i_beta_component,
    istell,
    beta_total_vol_avg,
    beta_thermal_vol_avg,
    beta_beam,
    beta_toroidal_vol_avg,
    beta_vol_avg_max,
):
    """Call PROCESS's `constraint_equation_24` through the port's signature."""
    data = DataStructure()
    data.physics.i_beta_component = i_beta_component
    data.stellarator.istell = istell
    data.physics.beta_total_vol_avg = beta_total_vol_avg
    data.physics.beta_thermal_vol_avg = beta_thermal_vol_avg
    data.physics.beta_beam = beta_beam
    data.physics.beta_toroidal_vol_avg = beta_toroidal_vol_avg
    data.physics.beta_vol_avg_max = beta_vol_avg_max
    return _evaluate(24, data)


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

    samples = [
        legacy_sample(
            "total",
            i_beta_component=int(BetaComponentLimits.TOTAL),
            istell=0,
            **_common,
        ),
        legacy_sample(
            "thermal",
            i_beta_component=int(BetaComponentLimits.THERMAL),
            istell=0,
            **_common,
        ),
        legacy_sample(
            "thermal-and-beam",
            i_beta_component=int(BetaComponentLimits.THERMAL_AND_BEAM),
            istell=0,
            **_common,
        ),
        legacy_sample(
            "toroidal",
            i_beta_component=int(BetaComponentLimits.TOROIDAL),
            istell=0,
            **_common,
        ),
        # istell != 0 overrides i_beta_component and always uses the TOTAL branch --
        # see the audit record's "real PROCESS finding" note. TOROIDAL is chosen here
        # deliberately (not TOTAL) so this sample would fail if the port ever stopped
        # honouring that override.
        legacy_sample(
            "stellarator-overrides-to-total",
            i_beta_component=int(BetaComponentLimits.TOROIDAL),
            istell=1,
            **_common,
        ),
    ]

    fuzz_bounds = {
        **bounds_from_iteration_variables("beta_total_vol_avg"),
        "beta_thermal_vol_avg": (0.001, 0.05),
        "beta_beam": (0.0, 0.01),
        "beta_toroidal_vol_avg": (0.001, 0.05),
        "beta_vol_avg_max": (0.01, 0.1),
    }
    fuzz_fixed = {"i_beta_component": int(BetaComponentLimits.TOTAL), "istell": 0}


def _reference_constraint_25(b_tf_inboard_peak_with_ripple, b_tf_inboard_max):
    data = DataStructure()
    data.tfcoil.b_tf_inboard_peak_with_ripple = b_tf_inboard_peak_with_ripple
    data.constraints.b_tf_inboard_max = b_tf_inboard_max
    return _evaluate(25, data)


class TestConstraint25(Tier1Contract):
    """`constraint_equation_25` -> `constraint_25`."""

    audit_record = "core/solver/constraints.md"
    reference = _reference_constraint_25
    ported = constraint_25

    samples = [
        legacy_sample(
            "feasible", b_tf_inboard_peak_with_ripple=10.0, b_tf_inboard_max=12.0
        ),
        legacy_sample(
            "infeasible", b_tf_inboard_peak_with_ripple=13.0, b_tf_inboard_max=12.0
        ),
    ]

    fuzz_bounds = {
        "b_tf_inboard_peak_with_ripple": (0.1, 30.0),
        "b_tf_inboard_max": (0.1, 30.0),
    }


def _reference_constraint_26(j_cs_flat_top_end, j_cs_critical_flat_top_end, fjohc):
    data = DataStructure()
    data.pf_coil.j_cs_flat_top_end = j_cs_flat_top_end
    data.pf_coil.j_cs_critical_flat_top_end = j_cs_critical_flat_top_end
    data.constraints.fjohc = fjohc
    return _evaluate(26, data)


class TestConstraint26(Tier1Contract):
    """`constraint_equation_26` -> `constraint_26`."""

    audit_record = "core/solver/constraints.md"
    reference = _reference_constraint_26
    ported = constraint_26

    samples = [
        legacy_sample(
            "feasible",
            j_cs_flat_top_end=1.5e7,
            j_cs_critical_flat_top_end=2.5e7,
            fjohc=0.7,
        ),
        legacy_sample(
            "infeasible",
            j_cs_flat_top_end=2.4e7,
            j_cs_critical_flat_top_end=2.5e7,
            fjohc=0.7,
        ),
    ]

    fuzz_bounds = {
        "j_cs_flat_top_end": (1.0e6, 5.0e7),
        "j_cs_critical_flat_top_end": (1.0e6, 5.0e7),
        "fjohc": (0.1, 1.0),
    }


def _reference_constraint_27(j_cs_pulse_start, j_cs_critical_pulse_start, fjohc0):
    data = DataStructure()
    data.pf_coil.j_cs_pulse_start = j_cs_pulse_start
    data.pf_coil.j_cs_critical_pulse_start = j_cs_critical_pulse_start
    data.constraints.fjohc0 = fjohc0
    return _evaluate(27, data)


class TestConstraint27(Tier1Contract):
    """`constraint_equation_27` -> `constraint_27`."""

    audit_record = "core/solver/constraints.md"
    reference = _reference_constraint_27
    ported = constraint_27

    samples = [
        legacy_sample(
            "feasible",
            j_cs_pulse_start=1.5e7,
            j_cs_critical_pulse_start=2.5e7,
            fjohc0=0.7,
        ),
        legacy_sample(
            "infeasible",
            j_cs_pulse_start=2.4e7,
            j_cs_critical_pulse_start=2.5e7,
            fjohc0=0.7,
        ),
    ]

    fuzz_bounds = {
        "j_cs_pulse_start": (1.0e6, 5.0e7),
        "j_cs_critical_pulse_start": (1.0e6, 5.0e7),
        "fjohc0": (0.1, 1.0),
    }


def _reference_constraint_28(i_plasma_ignited, big_q_plasma, big_q_plasma_min):
    data = DataStructure()
    data.physics.i_plasma_ignited = i_plasma_ignited
    data.current_drive.big_q_plasma = big_q_plasma
    data.constraints.big_q_plasma_min = big_q_plasma_min
    return _evaluate(28, data)


class TestConstraint28(Tier1Contract):
    """`constraint_equation_28` -> `constraint_28`."""

    audit_record = "core/solver/constraints.md"
    reference = _reference_constraint_28
    ported = constraint_28

    static_argnames = ("i_plasma_ignited",)

    samples = [
        legacy_sample(
            "feasible",
            i_plasma_ignited=int(PlasmaIgnitionModel.NON_IGNITED),
            big_q_plasma=15.0,
            big_q_plasma_min=10.0,
        ),
        legacy_sample(
            "infeasible",
            i_plasma_ignited=int(PlasmaIgnitionModel.NON_IGNITED),
            big_q_plasma=5.0,
            big_q_plasma_min=10.0,
        ),
    ]

    fuzz_bounds = {
        "big_q_plasma": (0.1, 100.0),
        "big_q_plasma_min": (1.0, 50.0),
    }
    fuzz_fixed = {"i_plasma_ignited": int(PlasmaIgnitionModel.NON_IGNITED)}


def test_constraint_28_ported_raises_when_ignited():
    """Constraint 28 is not valid for an ignited plasma -- the port must raise."""
    with pytest.raises(ValueError, match="i_plasma_ignited"):
        constraint_28(int(PlasmaIgnitionModel.IGNITED), 15.0, 10.0)


def test_constraint_28_reference_raises_when_ignited():
    """Same precondition, checked against PROCESS's own registered function."""
    with pytest.raises(ProcessValueError):
        _reference_constraint_28(int(PlasmaIgnitionModel.IGNITED), 15.0, 10.0)


def _reference_constraint_29(rmajor, rminor, rinboard):
    data = DataStructure()
    data.physics.rmajor = rmajor
    data.physics.rminor = rminor
    data.build.rinboard = rinboard
    return _evaluate(29, data)


class TestConstraint29(Tier1Contract):
    """`constraint_equation_29` -> `constraint_29`."""

    audit_record = "core/solver/constraints.md"
    reference = _reference_constraint_29
    ported = constraint_29

    samples = [
        legacy_sample("consistent", rmajor=8.0, rminor=2.5, rinboard=5.5),
        legacy_sample("inconsistent", rmajor=8.0, rminor=2.5, rinboard=5.0),
    ]

    fuzz_bounds = {
        "rmajor": (1.0, 20.0),
        "rminor": (0.1, 5.0),
        "rinboard": (0.1, 15.0),
    }


def _reference_constraint_30(p_hcd_injected_total_mw, p_hcd_injected_max):
    data = DataStructure()
    data.current_drive.p_hcd_injected_total_mw = p_hcd_injected_total_mw
    data.current_drive.p_hcd_injected_max = p_hcd_injected_max
    return _evaluate(30, data)


class TestConstraint30(Tier1Contract):
    """`constraint_equation_30` -> `constraint_30`."""

    audit_record = "core/solver/constraints.md"
    reference = _reference_constraint_30
    ported = constraint_30

    samples = [
        legacy_sample(
            "feasible", p_hcd_injected_total_mw=80.0, p_hcd_injected_max=150.0
        ),
        legacy_sample(
            "infeasible", p_hcd_injected_total_mw=200.0, p_hcd_injected_max=150.0
        ),
    ]

    fuzz_bounds = {
        "p_hcd_injected_total_mw": (0.1, 300.0),
        "p_hcd_injected_max": (10.0, 500.0),
    }


def _reference_constraint_31(sig_tf_case, sig_tf_case_max):
    data = DataStructure()
    data.tfcoil.sig_tf_case = sig_tf_case
    data.tfcoil.sig_tf_case_max = sig_tf_case_max
    return _evaluate(31, data)


class TestConstraint31(Tier1Contract):
    """`constraint_equation_31` -> `constraint_31`.

    See `batch3.md`/`constraint_31`'s own docstring: `sig_tf_case` is never populated
    by a real stellarator run (its only PROCESS producer, `tfcoil/superconducting.py`,
    is never called when `istell != 0`). Ported and tested as a pure arithmetic
    function regardless -- the port faithfully reproduces PROCESS's formula, the hole
    is in what data ever reaches it on a real run, not in this function.
    """

    audit_record = "core/solver/constraints.md"
    reference = _reference_constraint_31
    ported = constraint_31

    samples = [
        legacy_sample("feasible", sig_tf_case=3.0e8, sig_tf_case_max=6.0e8),
        legacy_sample("infeasible", sig_tf_case=7.0e8, sig_tf_case_max=6.0e8),
        # The value every real stellarator run actually has: sig_tf_case stuck at its
        # DataStructure default (0.0), always trivially feasible.
        legacy_sample("stellarator-default", sig_tf_case=0.0, sig_tf_case_max=6.0e8),
    ]

    fuzz_bounds = {
        "sig_tf_case": (0.0, 1.0e9),
        "sig_tf_case_max": (1.0e8, 1.0e9),
    }


def _reference_constraint_32(sig_tf_wp, sig_tf_wp_max):
    data = DataStructure()
    data.tfcoil.sig_tf_wp = sig_tf_wp
    data.tfcoil.sig_tf_wp_max = sig_tf_wp_max
    return _evaluate(32, data)


class TestConstraint32(Tier1Contract):
    """`constraint_equation_32` -> `constraint_32`."""

    audit_record = "core/solver/constraints.md"
    reference = _reference_constraint_32
    ported = constraint_32

    samples = [
        legacy_sample("feasible", sig_tf_wp=3.0e8, sig_tf_wp_max=6.0e8),
        legacy_sample("infeasible", sig_tf_wp=7.0e8, sig_tf_wp_max=6.0e8),
    ]

    fuzz_bounds = {
        "sig_tf_wp": (1.0e7, 1.0e9),
        "sig_tf_wp_max": (1.0e8, 1.0e9),
    }


def _reference_constraint_33(j_tf_wp, j_tf_wp_critical, f_j_tf_wp_critical_max):
    data = DataStructure()
    data.tfcoil.j_tf_wp = j_tf_wp
    data.tfcoil.j_tf_wp_critical = j_tf_wp_critical
    data.constraints.f_j_tf_wp_critical_max = f_j_tf_wp_critical_max
    return _evaluate(33, data)


class TestConstraint33(Tier1Contract):
    """`constraint_equation_33` -> `constraint_33`.

    See `batch3.md`/`constraint_33`'s own docstring: `j_tf_wp_critical` has the same
    "never populated on a real stellarator run" hole as constraint 31's `sig_tf_case`.
    Sample values here are still nonzero (unlike constraint 31's dedicated
    `stellarator-default` sample) since `j_tf_wp_critical == 0.0` would make the bound
    identically zero and every `fuzz` sample's `normalised_residual` a division by
    zero -- the pure function is tested faithfully across its domain regardless of
    which of that domain a real stellarator run ever visits.
    """

    audit_record = "core/solver/constraints.md"
    reference = _reference_constraint_33
    ported = constraint_33

    samples = [
        legacy_sample(
            "feasible",
            j_tf_wp=3.0e8,
            j_tf_wp_critical=6.0e8,
            f_j_tf_wp_critical_max=0.8,
        ),
        legacy_sample(
            "infeasible",
            j_tf_wp=5.5e8,
            j_tf_wp_critical=6.0e8,
            f_j_tf_wp_critical_max=0.8,
        ),
    ]

    fuzz_bounds = {
        "j_tf_wp": (1.0e7, 1.0e9),
        "j_tf_wp_critical": (1.0e8, 1.0e9),
        "f_j_tf_wp_critical_max": (0.1, 1.0),
    }


def _reference_constraint_34(v_tf_coil_dump_quench_kv, v_tf_coil_dump_quench_max_kv):
    data = DataStructure()
    data.tfcoil.v_tf_coil_dump_quench_kv = v_tf_coil_dump_quench_kv
    data.tfcoil.v_tf_coil_dump_quench_max_kv = v_tf_coil_dump_quench_max_kv
    return _evaluate(34, data)


class TestConstraint34(Tier1Contract):
    """`constraint_equation_34` -> `constraint_34`."""

    audit_record = "core/solver/constraints.md"
    reference = _reference_constraint_34
    ported = constraint_34

    samples = [
        legacy_sample(
            "feasible",
            v_tf_coil_dump_quench_kv=5.0,
            v_tf_coil_dump_quench_max_kv=10.0,
        ),
        legacy_sample(
            "infeasible",
            v_tf_coil_dump_quench_kv=15.0,
            v_tf_coil_dump_quench_max_kv=10.0,
        ),
    ]

    fuzz_bounds = {
        "v_tf_coil_dump_quench_kv": (0.1, 30.0),
        "v_tf_coil_dump_quench_max_kv": (1.0, 30.0),
    }


def _reference_constraint_35(j_tf_wp, j_tf_wp_quench_heat_max):
    data = DataStructure()
    data.tfcoil.j_tf_wp = j_tf_wp
    data.tfcoil.j_tf_wp_quench_heat_max = j_tf_wp_quench_heat_max
    return _evaluate(35, data)


class TestConstraint35(Tier1Contract):
    """`constraint_equation_35` -> `constraint_35`."""

    audit_record = "core/solver/constraints.md"
    reference = _reference_constraint_35
    ported = constraint_35

    samples = [
        legacy_sample("feasible", j_tf_wp=3.0e8, j_tf_wp_quench_heat_max=6.0e8),
        legacy_sample("infeasible", j_tf_wp=7.0e8, j_tf_wp_quench_heat_max=6.0e8),
    ]

    fuzz_bounds = {
        "j_tf_wp": (1.0e7, 1.0e9),
        "j_tf_wp_quench_heat_max": (1.0e8, 1.0e9),
    }


def _reference_constraint_36(
    temp_tf_superconductor_margin, temp_tf_superconductor_margin_min
):
    data = DataStructure()
    data.tfcoil.temp_tf_superconductor_margin = temp_tf_superconductor_margin
    data.tfcoil.temp_tf_superconductor_margin_min = temp_tf_superconductor_margin_min
    return _evaluate(36, data)


class TestConstraint36(Tier1Contract):
    """`constraint_equation_36` -> `constraint_36`."""

    audit_record = "core/solver/constraints.md"
    reference = _reference_constraint_36
    ported = constraint_36

    samples = [
        legacy_sample(
            "feasible",
            temp_tf_superconductor_margin=1.5,
            temp_tf_superconductor_margin_min=1.0,
        ),
        legacy_sample(
            "infeasible",
            temp_tf_superconductor_margin=0.5,
            temp_tf_superconductor_margin_min=1.0,
        ),
    ]

    fuzz_bounds = {
        "temp_tf_superconductor_margin": (0.01, 5.0),
        "temp_tf_superconductor_margin_min": (0.1, 3.0),
    }


def _reference_constraint_37(eta_cd_norm_hcd_primary, eta_cd_norm_hcd_primary_max):
    data = DataStructure()
    data.current_drive.eta_cd_norm_hcd_primary = eta_cd_norm_hcd_primary
    data.constraints.eta_cd_norm_hcd_primary_max = eta_cd_norm_hcd_primary_max
    return _evaluate(37, data)


class TestConstraint37(Tier1Contract):
    """`constraint_equation_37` -> `constraint_37`."""

    audit_record = "core/solver/constraints.md"
    reference = _reference_constraint_37
    ported = constraint_37

    samples = [
        legacy_sample(
            "feasible", eta_cd_norm_hcd_primary=0.3, eta_cd_norm_hcd_primary_max=0.5
        ),
        legacy_sample(
            "infeasible", eta_cd_norm_hcd_primary=0.7, eta_cd_norm_hcd_primary_max=0.5
        ),
    ]

    fuzz_bounds = {
        "eta_cd_norm_hcd_primary": (0.01, 1.0),
        "eta_cd_norm_hcd_primary_max": (0.1, 2.0),
    }


def _reference_constraint_39(temp_fw_peak, temp_fw_max):
    data = DataStructure()
    data.fwbs.temp_fw_peak = temp_fw_peak
    data.fwbs.temp_fw_max = temp_fw_max
    return _evaluate(39, data)


class TestConstraint39(Tier1Contract):
    """`constraint_equation_39` -> `constraint_39`."""

    audit_record = "core/solver/constraints.md"
    reference = _reference_constraint_39
    ported = constraint_39

    samples = [
        legacy_sample("feasible", temp_fw_peak=550.0, temp_fw_max=650.0),
        legacy_sample("infeasible", temp_fw_peak=700.0, temp_fw_max=650.0),
    ]

    fuzz_bounds = {
        "temp_fw_peak": (300.0, 900.0),
        "temp_fw_max": (400.0, 1000.0),
    }


def _reference_constraint_40(p_hcd_injected_total_mw, p_hcd_injected_min_mw):
    data = DataStructure()
    data.current_drive.p_hcd_injected_total_mw = p_hcd_injected_total_mw
    data.constraints.p_hcd_injected_min_mw = p_hcd_injected_min_mw
    return _evaluate(40, data)


class TestConstraint40(Tier1Contract):
    """`constraint_equation_40` -> `constraint_40`."""

    audit_record = "core/solver/constraints.md"
    reference = _reference_constraint_40
    ported = constraint_40

    samples = [
        legacy_sample(
            "feasible", p_hcd_injected_total_mw=60.0, p_hcd_injected_min_mw=50.0
        ),
        legacy_sample(
            "infeasible", p_hcd_injected_total_mw=40.0, p_hcd_injected_min_mw=50.0
        ),
    ]

    fuzz_bounds = {
        "p_hcd_injected_total_mw": (0.0, 500.0),
        "p_hcd_injected_min_mw": (1.0, 200.0),
    }


def _reference_constraint_41(
    t_plant_pulse_plasma_current_ramp_up, t_current_ramp_up_min
):
    data = DataStructure()
    data.times.t_plant_pulse_plasma_current_ramp_up = (
        t_plant_pulse_plasma_current_ramp_up
    )
    data.constraints.t_current_ramp_up_min = t_current_ramp_up_min
    return _evaluate(41, data)


class TestConstraint41(Tier1Contract):
    """`constraint_equation_41` -> `constraint_41`."""

    audit_record = "core/solver/constraints.md"
    reference = _reference_constraint_41
    ported = constraint_41

    samples = [
        legacy_sample(
            "feasible",
            t_plant_pulse_plasma_current_ramp_up=15.0,
            t_current_ramp_up_min=10.0,
        ),
        legacy_sample(
            "infeasible",
            t_plant_pulse_plasma_current_ramp_up=5.0,
            t_current_ramp_up_min=10.0,
        ),
    ]

    fuzz_bounds = {
        **bounds_from_iteration_variables("t_plant_pulse_plasma_current_ramp_up"),
        "t_current_ramp_up_min": (1.0, 500.0),
    }


def _reference_constraint_42(t_plant_pulse_total, t_cycle_min):
    data = DataStructure()
    data.times.t_plant_pulse_total = t_plant_pulse_total
    data.constraints.t_cycle_min = t_cycle_min
    return _evaluate(42, data)


class TestConstraint42(Tier1Contract):
    """`constraint_equation_42` -> `constraint_42`."""

    audit_record = "core/solver/constraints.md"
    reference = _reference_constraint_42
    ported = constraint_42

    samples = [
        legacy_sample("feasible", t_plant_pulse_total=2000.0, t_cycle_min=1800.0),
        legacy_sample("infeasible", t_plant_pulse_total=1500.0, t_cycle_min=1800.0),
    ]

    fuzz_bounds = {
        "t_plant_pulse_total": (100.0, 1.0e4),
        "t_cycle_min": (100.0, 1.0e4),
    }


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

    samples = [
        legacy_sample(
            "superconducting-consistent",
            i_tf_sup=int(TFConductorModel.SUPERCONDUCTING),
            temp_cp_average=350.0,
            tcpav2=350.0,
        ),
        legacy_sample(
            "water-cooled-copper-consistent",
            i_tf_sup=int(TFConductorModel.WATER_COOLED_COPPER),
            temp_cp_average=350.0,
            tcpav2=350.0,
        ),
        legacy_sample(
            "inconsistent",
            i_tf_sup=int(TFConductorModel.SUPERCONDUCTING),
            temp_cp_average=350.0,
            tcpav2=360.0,
        ),
    ]

    fuzz_bounds = {
        "temp_cp_average": (250.0, 600.0),
        "tcpav2": (250.0, 600.0),
    }
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

    samples = [
        legacy_sample(
            "superconducting-feasible",
            i_tf_sup=int(TFConductorModel.SUPERCONDUCTING),
            temp_cp_max=650.0,
            temp_cp_peak=600.0,
        ),
        legacy_sample(
            "water-cooled-copper-feasible",
            i_tf_sup=int(TFConductorModel.WATER_COOLED_COPPER),
            temp_cp_max=650.0,
            temp_cp_peak=600.0,
        ),
        legacy_sample(
            "infeasible",
            i_tf_sup=int(TFConductorModel.SUPERCONDUCTING),
            temp_cp_max=650.0,
            temp_cp_peak=700.0,
        ),
    ]

    fuzz_bounds = {
        "temp_cp_max": (400.0, 1000.0),
        "temp_cp_peak": (300.0, 1000.0),
    }
    fuzz_fixed = {"i_tf_sup": int(TFConductorModel.SUPERCONDUCTING)}


def _reference_constraint_45(itart, q95, q95_min):
    """Call PROCESS's `constraint_manager_45` through the port's signature."""
    data = DataStructure()
    data.physics.itart = itart
    data.physics.q95 = q95
    data.physics.q95_min = q95_min
    return _evaluate(45, data)


class TestConstraint45(Tier1Contract):
    """`constraint_manager_45` -> `constraint_45`."""

    audit_record = "core/solver/constraints.md"
    reference = _reference_constraint_45
    ported = constraint_45

    static_argnames = ("itart",)

    samples = [
        legacy_sample("feasible", itart=1, q95=3.5, q95_min=3.0),
        legacy_sample("infeasible", itart=1, q95=2.5, q95_min=3.0),
    ]

    fuzz_bounds = {"q95": (1.5, 10.0), "q95_min": (1.0, 5.0)}
    fuzz_fixed = {"itart": 1}


def _reference_constraint_46(itart, eps, plasma_current, c_tf_total):
    """Call PROCESS's `constraint_equation_46` through the port's signature."""
    data = DataStructure()
    data.physics.itart = itart
    data.physics.eps = eps
    data.physics.plasma_current = plasma_current
    data.tfcoil.c_tf_total = c_tf_total
    return _evaluate(46, data)


class TestConstraint46(Tier1Contract):
    """`constraint_equation_46` -> `constraint_46`."""

    audit_record = "core/solver/constraints.md"
    reference = _reference_constraint_46
    ported = constraint_46

    static_argnames = ("itart",)

    samples = [
        legacy_sample(
            "feasible", itart=1, eps=0.7, plasma_current=1.0e7, c_tf_total=3.0e7
        ),
        legacy_sample(
            "infeasible", itart=1, eps=0.9, plasma_current=1.0e7, c_tf_total=5.0e6
        ),
    ]

    fuzz_bounds = {
        "eps": (0.4, 0.95),
        "plasma_current": (1.0e6, 5.0e7),
        "c_tf_total": (1.0e6, 1.0e8),
    }
    fuzz_fixed = {"itart": 1}


def _reference_constraint_48(beta_poloidal_vol_avg, beta_poloidal_max):
    """Call PROCESS's `constraint_equation_48` through the port's signature."""
    data = DataStructure()
    data.physics.beta_poloidal_vol_avg = beta_poloidal_vol_avg
    data.constraints.beta_poloidal_max = beta_poloidal_max
    return _evaluate(48, data)


class TestConstraint48(Tier1Contract):
    """`constraint_equation_48` -> `constraint_48`."""

    audit_record = "core/solver/constraints.md"
    reference = _reference_constraint_48
    ported = constraint_48

    samples = [
        legacy_sample("feasible", beta_poloidal_vol_avg=0.5, beta_poloidal_max=1.0),
        legacy_sample("infeasible", beta_poloidal_vol_avg=1.2, beta_poloidal_max=1.0),
    ]

    fuzz_bounds = {
        "beta_poloidal_vol_avg": (0.01, 2.0),
        "beta_poloidal_max": (0.1, 3.0),
    }


def _reference_constraint_51(vs_plasma_ramp_required, vs_cs_pf_total_ramp):
    """Call PROCESS's `constraint_equation_51` through the port's signature."""
    data = DataStructure()
    data.physics.vs_plasma_ramp_required = vs_plasma_ramp_required
    data.pf_coil.vs_cs_pf_total_ramp = vs_cs_pf_total_ramp
    return _evaluate(51, data)


class TestConstraint51(Tier1Contract):
    """`constraint_equation_51` -> `constraint_51`."""

    audit_record = "core/solver/constraints.md"
    reference = _reference_constraint_51
    ported = constraint_51

    samples = [
        legacy_sample(
            "matched", vs_plasma_ramp_required=-120.0, vs_cs_pf_total_ramp=120.0
        ),
        legacy_sample(
            "mismatched", vs_plasma_ramp_required=-100.0, vs_cs_pf_total_ramp=120.0
        ),
    ]

    fuzz_bounds = {
        "vs_plasma_ramp_required": (-300.0, 300.0),
        "vs_cs_pf_total_ramp": (1.0, 300.0),
    }


def _reference_constraint_53(flu_tf_neutron_fast_peak, flu_tf_neutron_fast_max):
    """Call PROCESS's `constraint_equation_53` through the port's signature."""
    data = DataStructure()
    data.fwbs.flu_tf_neutron_fast_peak = flu_tf_neutron_fast_peak
    data.constraints.flu_tf_neutron_fast_max = flu_tf_neutron_fast_max
    return _evaluate(53, data)


class TestConstraint53(Tier1Contract):
    """`constraint_equation_53` -> `constraint_53`."""

    audit_record = "core/solver/constraints.md"
    reference = _reference_constraint_53
    ported = constraint_53

    samples = [
        legacy_sample(
            "feasible",
            flu_tf_neutron_fast_peak=5.0e22,
            flu_tf_neutron_fast_max=1.0e23,
        ),
        legacy_sample(
            "infeasible",
            flu_tf_neutron_fast_peak=1.5e23,
            flu_tf_neutron_fast_max=1.0e23,
        ),
    ]

    fuzz_bounds = {
        "flu_tf_neutron_fast_peak": (1.0e20, 5.0e23),
        "flu_tf_neutron_fast_max": (1.0e21, 2.0e23),
    }


def _reference_constraint_54(ptfnucpm3, ptfnucmax):
    """Call PROCESS's `constraint_equation_54` through the port's signature."""
    data = DataStructure()
    data.fwbs.ptfnucpm3 = ptfnucpm3
    data.constraints.ptfnucmax = ptfnucmax
    return _evaluate(54, data)


class TestConstraint54(Tier1Contract):
    """`constraint_equation_54` -> `constraint_54`."""

    audit_record = "core/solver/constraints.md"
    reference = _reference_constraint_54
    ported = constraint_54

    samples = [
        legacy_sample("feasible", ptfnucpm3=5.0e-4, ptfnucmax=1.0e-3),
        legacy_sample("infeasible", ptfnucpm3=2.0e-3, ptfnucmax=1.0e-3),
    ]

    fuzz_bounds = {"ptfnucpm3": (1.0e-5, 5.0e-3), "ptfnucmax": (1.0e-4, 5.0e-3)}


def _reference_constraint_56(
    p_plasma_separatrix_rmajor_mw, p_plasma_separatrix_rmajor_max_mw
):
    data = DataStructure()
    data.physics.p_plasma_separatrix_rmajor_mw = p_plasma_separatrix_rmajor_mw
    data.constraints.p_plasma_separatrix_rmajor_max_mw = (
        p_plasma_separatrix_rmajor_max_mw
    )
    return _evaluate(56, data)


class TestConstraint56(Tier1Contract):
    """`constraint_equation_56` -> `constraint_56`."""

    audit_record = "core/solver/constraints.md"
    reference = _reference_constraint_56
    ported = constraint_56

    samples = [
        legacy_sample(
            "feasible",
            p_plasma_separatrix_rmajor_mw=8.0,
            p_plasma_separatrix_rmajor_max_mw=17.0,
        ),
        legacy_sample(
            "infeasible",
            p_plasma_separatrix_rmajor_mw=20.0,
            p_plasma_separatrix_rmajor_max_mw=17.0,
        ),
    ]

    fuzz_bounds = {
        "p_plasma_separatrix_rmajor_mw": (0.1, 30.0),
        "p_plasma_separatrix_rmajor_max_mw": (1.0, 30.0),
    }


def _reference_constraint_59(f_p_beam_shine_through, f_p_beam_shine_through_max):
    data = DataStructure()
    data.current_drive.f_p_beam_shine_through = f_p_beam_shine_through
    data.constraints.f_p_beam_shine_through_max = f_p_beam_shine_through_max
    return _evaluate(59, data)


class TestConstraint59(Tier1Contract):
    """`constraint_equation_59` -> `constraint_59`."""

    audit_record = "core/solver/constraints.md"
    reference = _reference_constraint_59
    ported = constraint_59

    samples = [
        legacy_sample(
            "feasible", f_p_beam_shine_through=0.005, f_p_beam_shine_through_max=0.01
        ),
        legacy_sample(
            "infeasible", f_p_beam_shine_through=0.02, f_p_beam_shine_through_max=0.01
        ),
    ]

    fuzz_bounds = {
        "f_p_beam_shine_through": (0.0, 0.1),
        "f_p_beam_shine_through_max": (0.001, 0.1),
    }


def _reference_constraint_60(
    temp_cs_superconductor_margin, temp_cs_superconductor_margin_min
):
    data = DataStructure()
    data.pf_coil.temp_cs_superconductor_margin = temp_cs_superconductor_margin
    data.tfcoil.temp_cs_superconductor_margin_min = temp_cs_superconductor_margin_min
    return _evaluate(60, data)


class TestConstraint60(Tier1Contract):
    """`constraint_equation_60` -> `constraint_60`."""

    audit_record = "core/solver/constraints.md"
    reference = _reference_constraint_60
    ported = constraint_60

    samples = [
        legacy_sample(
            "feasible",
            temp_cs_superconductor_margin=2.0,
            temp_cs_superconductor_margin_min=1.5,
        ),
        legacy_sample(
            "infeasible",
            temp_cs_superconductor_margin=1.0,
            temp_cs_superconductor_margin_min=1.5,
        ),
    ]

    fuzz_bounds = {
        "temp_cs_superconductor_margin": (0.1, 5.0),
        "temp_cs_superconductor_margin_min": (0.1, 3.0),
    }


def _reference_constraint_61(f_t_plant_available, f_t_plant_available_min):
    data = DataStructure()
    data.costs.f_t_plant_available = f_t_plant_available
    data.costs.f_t_plant_available_min = f_t_plant_available_min
    return _evaluate(61, data)


class TestConstraint61(Tier1Contract):
    """`constraint_equation_61` -> `constraint_61`."""

    audit_record = "core/solver/constraints.md"
    reference = _reference_constraint_61
    ported = constraint_61

    samples = [
        legacy_sample(
            "feasible", f_t_plant_available=0.85, f_t_plant_available_min=0.75
        ),
        legacy_sample(
            "infeasible", f_t_plant_available=0.6, f_t_plant_available_min=0.75
        ),
    ]

    fuzz_bounds = {
        "f_t_plant_available": (0.1, 1.0),
        "f_t_plant_available_min": (0.1, 1.0),
    }


def _reference_constraint_62(
    f_t_alpha_energy_confinement, f_t_alpha_energy_confinement_min
):
    data = DataStructure()
    data.physics.f_t_alpha_energy_confinement = f_t_alpha_energy_confinement
    data.constraints.f_t_alpha_energy_confinement_min = f_t_alpha_energy_confinement_min
    return _evaluate(62, data)


class TestConstraint62(Tier1Contract):
    """`constraint_equation_62` -> `constraint_62`."""

    audit_record = "core/solver/constraints.md"
    reference = _reference_constraint_62
    ported = constraint_62

    samples = [
        legacy_sample(
            "feasible",
            f_t_alpha_energy_confinement=6.0,
            f_t_alpha_energy_confinement_min=5.0,
        ),
        legacy_sample(
            "infeasible",
            f_t_alpha_energy_confinement=4.0,
            f_t_alpha_energy_confinement_min=5.0,
        ),
    ]

    fuzz_bounds = {
        "f_t_alpha_energy_confinement": (0.1, 20.0),
        "f_t_alpha_energy_confinement_min": (0.1, 10.0),
    }


def _reference_constraint_63(n_iter_vacuum_pumps, n_tf_coils):
    data = DataStructure()
    data.vacuum.n_iter_vacuum_pumps = n_iter_vacuum_pumps
    data.tfcoil.n_tf_coils = n_tf_coils
    return _evaluate(63, data)


class TestConstraint63(Tier1Contract):
    """`constraint_equation_63` -> `constraint_63`."""

    audit_record = "core/solver/constraints.md"
    reference = _reference_constraint_63
    ported = constraint_63

    samples = [
        legacy_sample("feasible", n_iter_vacuum_pumps=30.0, n_tf_coils=50.0),
        legacy_sample("infeasible", n_iter_vacuum_pumps=60.0, n_tf_coils=50.0),
    ]

    fuzz_bounds = {
        "n_iter_vacuum_pumps": (1.0, 100.0),
        "n_tf_coils": (1.0, 100.0),
    }


def _reference_constraint_64(
    n_charge_plasma_effective_vol_avg, n_charge_plasma_effective_vol_avg_max
):
    data = DataStructure()
    data.physics.n_charge_plasma_effective_vol_avg = n_charge_plasma_effective_vol_avg
    data.constraints.n_charge_plasma_effective_vol_avg_max = (
        n_charge_plasma_effective_vol_avg_max
    )
    return _evaluate(64, data)


class TestConstraint64(Tier1Contract):
    """`constraint_equation_64` -> `constraint_64`."""

    audit_record = "core/solver/constraints.md"
    reference = _reference_constraint_64
    ported = constraint_64

    samples = [
        legacy_sample(
            "feasible",
            n_charge_plasma_effective_vol_avg=1.8,
            n_charge_plasma_effective_vol_avg_max=2.5,
        ),
        legacy_sample(
            "infeasible",
            n_charge_plasma_effective_vol_avg=3.0,
            n_charge_plasma_effective_vol_avg_max=2.5,
        ),
    ]

    fuzz_bounds = {
        "n_charge_plasma_effective_vol_avg": (1.0, 5.0),
        "n_charge_plasma_effective_vol_avg_max": (1.0, 5.0),
    }


def _reference_constraint_65(vv_stress_quench, max_vv_stress):
    data = DataStructure()
    data.superconducting_tfcoil.vv_stress_quench = vv_stress_quench
    data.tfcoil.max_vv_stress = max_vv_stress
    return _evaluate(65, data)


class TestConstraint65(Tier1Contract):
    """`constraint_equation_65` -> `constraint_65`."""

    audit_record = "core/solver/constraints.md"
    reference = _reference_constraint_65
    ported = constraint_65

    samples = [
        legacy_sample("feasible", vv_stress_quench=1.5e8, max_vv_stress=2.0e8),
        legacy_sample("infeasible", vv_stress_quench=2.5e8, max_vv_stress=2.0e8),
    ]

    fuzz_bounds = {
        "vv_stress_quench": (1.0e6, 5.0e8),
        "max_vv_stress": (1.0e6, 5.0e8),
    }


def _reference_constraint_66(peakpoloidalpower, maxpoloidalpower):
    data = DataStructure()
    data.pf_power.peakpoloidalpower = peakpoloidalpower
    data.pf_power.maxpoloidalpower = maxpoloidalpower
    return _evaluate(66, data)


class TestConstraint66(Tier1Contract):
    """`constraint_equation_66` -> `constraint_66`."""

    audit_record = "core/solver/constraints.md"
    reference = _reference_constraint_66
    ported = constraint_66
    samples = [
        legacy_sample("typical", peakpoloidalpower=150.0, maxpoloidalpower=300.0),
        legacy_sample("near-limit", peakpoloidalpower=295.0, maxpoloidalpower=300.0),
    ]


def _reference_constraint_67(pflux_fw_rad_max_mw, pflux_fw_rad_max):
    data = DataStructure()
    data.constraints.pflux_fw_rad_max_mw = pflux_fw_rad_max_mw
    data.constraints.pflux_fw_rad_max = pflux_fw_rad_max
    return _evaluate(67, data)


class TestConstraint67(Tier1Contract):
    """`constraint_equation_67` -> `constraint_67`."""

    audit_record = "core/solver/constraints.md"
    reference = _reference_constraint_67
    ported = constraint_67
    samples = [
        legacy_sample("typical", pflux_fw_rad_max_mw=0.3, pflux_fw_rad_max=0.5),
        legacy_sample("near-limit", pflux_fw_rad_max_mw=0.49, pflux_fw_rad_max=0.5),
    ]


def _reference_constraint_68(
    i_q95_fixed,
    p_plasma_separatrix_mw,
    b_plasma_toroidal_on_axis,
    q95,
    q95_fixed,
    aspect,
    rmajor,
    p_div_bt_q_aspect_rmajor_mw,
    p_div_bt_q_aspect_rmajor_max_mw,
):
    data = DataStructure()
    data.constraints.i_q95_fixed = i_q95_fixed
    data.physics.p_plasma_separatrix_mw = p_plasma_separatrix_mw
    data.physics.b_plasma_toroidal_on_axis = b_plasma_toroidal_on_axis
    data.physics.q95 = q95
    data.constraints.q95_fixed = q95_fixed
    data.physics.aspect = aspect
    data.physics.rmajor = rmajor
    data.physics.p_div_bt_q_aspect_rmajor_mw = p_div_bt_q_aspect_rmajor_mw
    data.constraints.p_div_bt_q_aspect_rmajor_max_mw = p_div_bt_q_aspect_rmajor_max_mw
    return _evaluate(68, data)


class TestConstraint68(Tier1Contract):
    """`constraint_equation_68` -> `constraint_68`."""

    audit_record = "core/solver/constraints.md"
    reference = _reference_constraint_68
    ported = constraint_68
    static_argnames = ("i_q95_fixed",)
    samples = [
        legacy_sample(
            "fixed-q95",
            i_q95_fixed=1,
            p_plasma_separatrix_mw=100.0,
            b_plasma_toroidal_on_axis=5.0,
            q95=3.5,
            q95_fixed=3.0,
            aspect=3.0,
            rmajor=8.0,
            p_div_bt_q_aspect_rmajor_mw=15.0,
            p_div_bt_q_aspect_rmajor_max_mw=20.0,
        ),
        legacy_sample(
            "free-q95",
            i_q95_fixed=0,
            p_plasma_separatrix_mw=100.0,
            b_plasma_toroidal_on_axis=5.0,
            q95=3.5,
            q95_fixed=3.0,
            aspect=3.0,
            rmajor=8.0,
            p_div_bt_q_aspect_rmajor_mw=15.0,
            p_div_bt_q_aspect_rmajor_max_mw=20.0,
        ),
    ]


def _reference_constraint_72(
    i_tf_bucking,
    i_tf_inside_cs,
    stress_shear_cs_peak,
    sig_tf_cs_bucked,
    stress_cs_steel_max,
):
    data = DataStructure()
    data.tfcoil.i_tf_bucking = i_tf_bucking
    data.build.i_tf_inside_cs = i_tf_inside_cs
    data.pf_coil.stress_shear_cs_peak = stress_shear_cs_peak
    data.tfcoil.sig_tf_cs_bucked = sig_tf_cs_bucked
    data.pf_coil.stress_cs_steel_max = stress_cs_steel_max
    return _evaluate(72, data)


class TestConstraint72(Tier1Contract):
    """`constraint_equation_72` -> `constraint_72`."""

    audit_record = "core/solver/constraints.md"
    reference = _reference_constraint_72
    ported = constraint_72
    static_argnames = ("i_tf_bucking", "i_tf_inside_cs")
    samples = [
        legacy_sample(
            "bucked-and-wedged",
            i_tf_bucking=2,
            i_tf_inside_cs=int(TFCSRadialConfiguration.TF_OUTSIDE_CS),
            stress_shear_cs_peak=500e6,
            sig_tf_cs_bucked=550e6,
            stress_cs_steel_max=660e6,
        ),
        legacy_sample(
            "free-standing",
            i_tf_bucking=1,
            i_tf_inside_cs=int(TFCSRadialConfiguration.TF_OUTSIDE_CS),
            stress_shear_cs_peak=500e6,
            sig_tf_cs_bucked=550e6,
            stress_cs_steel_max=660e6,
        ),
    ]


def _reference_constraint_73(
    p_plasma_separatrix_mw, p_l_h_threshold_mw, p_hcd_injected_total_mw
):
    data = DataStructure()
    data.physics.p_plasma_separatrix_mw = p_plasma_separatrix_mw
    data.physics.p_l_h_threshold_mw = p_l_h_threshold_mw
    data.current_drive.p_hcd_injected_total_mw = p_hcd_injected_total_mw
    return _evaluate(73, data)


class TestConstraint73(Tier1Contract):
    """`constraint_equation_73` -> `constraint_73`."""

    audit_record = "core/solver/constraints.md"
    reference = _reference_constraint_73
    ported = constraint_73
    samples = [
        legacy_sample(
            "typical",
            p_plasma_separatrix_mw=100.0,
            p_l_h_threshold_mw=60.0,
            p_hcd_injected_total_mw=30.0,
        ),
        legacy_sample(
            "near-limit",
            p_plasma_separatrix_mw=91.0,
            p_l_h_threshold_mw=60.0,
            p_hcd_injected_total_mw=30.0,
        ),
    ]


def _reference_constraint_74(temp_croco_quench, temp_croco_quench_max):
    data = DataStructure()
    data.tfcoil.temp_croco_quench = temp_croco_quench
    data.tfcoil.temp_croco_quench_max = temp_croco_quench_max
    return _evaluate(74, data)


class TestConstraint74(Tier1Contract):
    """`constraint_equation_74` -> `constraint_74`."""

    audit_record = "core/solver/constraints.md"
    reference = _reference_constraint_74
    ported = constraint_74
    samples = [
        legacy_sample("typical", temp_croco_quench=250.0, temp_croco_quench_max=300.0),
        legacy_sample(
            "near-limit", temp_croco_quench=298.0, temp_croco_quench_max=300.0
        ),
    ]


def _reference_constraint_75(coppera_m2, tf_coppera_m2_max):
    data = DataStructure()
    data.rebco.coppera_m2 = coppera_m2
    data.superconducting_tfcoil.tf_coppera_m2_max = tf_coppera_m2_max
    return _evaluate(75, data)


class TestConstraint75(Tier1Contract):
    """`constraint_equation_75` -> `constraint_75`."""

    audit_record = "core/solver/constraints.md"
    reference = _reference_constraint_75
    ported = constraint_75
    samples = [
        legacy_sample("typical", coppera_m2=1.0e8, tf_coppera_m2_max=2.0e8),
        legacy_sample("near-limit", coppera_m2=1.95e8, tf_coppera_m2_max=2.0e8),
    ]


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
    samples = [
        legacy_sample(
            "typical",
            kappa=1.7,
            triang=0.4,
            aspect=3.0,
            p_plasma_separatrix_mw=100.0,
            nd_plasma_electron_max_array_7=1.0e20,
            nd_plasma_separatrix_electron=5.0e19,
        ),
        legacy_sample(
            "high-elongation",
            kappa=2.2,
            triang=0.6,
            aspect=2.5,
            p_plasma_separatrix_mw=150.0,
            nd_plasma_electron_max_array_7=8.0e19,
            nd_plasma_separatrix_electron=6.0e19,
        ),
    ]


def _reference_constraint_77(c_tf_turn, c_tf_turn_max):
    data = DataStructure()
    data.tfcoil.c_tf_turn = c_tf_turn
    data.tfcoil.c_tf_turn_max = c_tf_turn_max
    return _evaluate(77, data)


class TestConstraint77(Tier1Contract):
    """`constraint_equation_77` -> `constraint_77`."""

    audit_record = "core/solver/constraints.md"
    reference = _reference_constraint_77
    ported = constraint_77

    samples = [
        legacy_sample("feasible", c_tf_turn=6.0e4, c_tf_turn_max=9.0e4),
        legacy_sample("infeasible", c_tf_turn=1.0e5, c_tf_turn_max=9.0e4),
    ]

    fuzz_bounds = {
        "c_tf_turn": (1.0e3, 2.0e5),
        "c_tf_turn_max": (1.0e3, 2.0e5),
    }


def _reference_constraint_78(fzactual, fzmin):
    data = DataStructure()
    data.reinke.fzactual = fzactual
    data.reinke.fzmin = fzmin
    return _evaluate(78, data)


class TestConstraint78(Tier1Contract):
    """`constraint_equation_78` -> `constraint_78`."""

    audit_record = "core/solver/constraints.md"
    reference = _reference_constraint_78
    ported = constraint_78

    samples = [
        legacy_sample("feasible", fzactual=5.0e-4, fzmin=3.0e-4),
        legacy_sample("infeasible", fzactual=1.0e-4, fzmin=3.0e-4),
    ]

    fuzz_bounds = {
        "fzactual": (1.0e-6, 1.0e-2),
        "fzmin": (1.0e-6, 1.0e-2),
    }


def _reference_constraint_79(
    b_cs_peak_flat_top_end, b_cs_peak_pulse_start, b_cs_limit_max
):
    data = DataStructure()
    data.pf_coil.b_cs_peak_flat_top_end = b_cs_peak_flat_top_end
    data.pf_coil.b_cs_peak_pulse_start = b_cs_peak_pulse_start
    data.pf_coil.b_cs_limit_max = b_cs_limit_max
    return _evaluate(79, data)


class TestConstraint79(Tier1Contract):
    """`constraint_equation_79` -> `constraint_79`."""

    audit_record = "core/solver/constraints.md"
    reference = _reference_constraint_79
    ported = constraint_79

    samples = [
        legacy_sample(
            "feasible-flattop-larger",
            b_cs_peak_flat_top_end=12.0,
            b_cs_peak_pulse_start=10.0,
            b_cs_limit_max=13.0,
        ),
        legacy_sample(
            "feasible-pulsestart-larger",
            b_cs_peak_flat_top_end=8.0,
            b_cs_peak_pulse_start=11.0,
            b_cs_limit_max=13.0,
        ),
        legacy_sample(
            "infeasible",
            b_cs_peak_flat_top_end=14.0,
            b_cs_peak_pulse_start=10.0,
            b_cs_limit_max=13.0,
        ),
    ]

    fuzz_bounds = {
        "b_cs_peak_flat_top_end": (1.0, 20.0),
        "b_cs_peak_pulse_start": (1.0, 20.0),
        "b_cs_limit_max": (1.0, 20.0),
    }


def _reference_constraint_80(p_plasma_separatrix_mw, p_plasma_separatrix_min_mw):
    data = DataStructure()
    data.physics.p_plasma_separatrix_mw = p_plasma_separatrix_mw
    data.constraints.p_plasma_separatrix_min_mw = p_plasma_separatrix_min_mw
    return _evaluate(80, data)


class TestConstraint80(Tier1Contract):
    """`constraint_equation_80` -> `constraint_80`."""

    audit_record = "core/solver/constraints.md"
    reference = _reference_constraint_80
    ported = constraint_80

    samples = [
        legacy_sample(
            "feasible",
            p_plasma_separatrix_mw=180.0,
            p_plasma_separatrix_min_mw=150.0,
        ),
        legacy_sample(
            "infeasible",
            p_plasma_separatrix_mw=100.0,
            p_plasma_separatrix_min_mw=150.0,
        ),
    ]

    fuzz_bounds = {
        "p_plasma_separatrix_mw": (1.0, 1000.0),
        "p_plasma_separatrix_min_mw": (1.0, 1000.0),
    }


def _reference_constraint_81(nd_plasma_electron_on_axis, nd_plasma_pedestal_electron):
    data = DataStructure()
    data.physics.nd_plasma_electron_on_axis = nd_plasma_electron_on_axis
    data.physics.nd_plasma_pedestal_electron = nd_plasma_pedestal_electron
    return _evaluate(81, data)


class TestConstraint81(Tier1Contract):
    """`constraint_equation_81` -> `constraint_81`."""

    audit_record = "core/solver/constraints.md"
    reference = _reference_constraint_81
    ported = constraint_81

    samples = [
        legacy_sample(
            "feasible",
            nd_plasma_electron_on_axis=1.1e20,
            nd_plasma_pedestal_electron=8.0e19,
        ),
        legacy_sample(
            "infeasible",
            nd_plasma_electron_on_axis=6.0e19,
            nd_plasma_pedestal_electron=8.0e19,
        ),
    ]

    fuzz_bounds = {
        "nd_plasma_electron_on_axis": (1.0e19, 3.0e20),
        "nd_plasma_pedestal_electron": (1.0e19, 3.0e20),
    }


def _reference_constraint_82(toroidalgap, dx_tf_inboard_out_toroidal):
    """Call PROCESS's `constraint_equation_82` through the port's signature."""
    data = DataStructure()
    data.tfcoil.toroidalgap = toroidalgap
    data.tfcoil.dx_tf_inboard_out_toroidal = dx_tf_inboard_out_toroidal
    return _evaluate(82, data)


class TestConstraint82(Tier1Contract):
    """`constraint_equation_82` -> `constraint_82`."""

    audit_record = "core/solver/constraints.md"
    reference = _reference_constraint_82
    ported = constraint_82

    samples = [
        legacy_sample("feasible", toroidalgap=0.5, dx_tf_inboard_out_toroidal=0.3),
        legacy_sample("infeasible", toroidalgap=0.2, dx_tf_inboard_out_toroidal=0.4),
    ]

    fuzz_bounds = {
        "toroidalgap": (0.01, 2.0),
        "dx_tf_inboard_out_toroidal": (0.01, 2.0),
    }


def _reference_constraint_83(available_radial_space, required_radial_space):
    """Call PROCESS's `constraint_equation_83` through the port's signature."""
    data = DataStructure()
    data.build.available_radial_space = available_radial_space
    data.build.required_radial_space = required_radial_space
    return _evaluate(83, data)


class TestConstraint83(Tier1Contract):
    """`constraint_equation_83` -> `constraint_83`."""

    audit_record = "core/solver/constraints.md"
    reference = _reference_constraint_83
    ported = constraint_83

    samples = [
        legacy_sample("feasible", available_radial_space=2.5, required_radial_space=2.0),
        legacy_sample(
            "infeasible", available_radial_space=1.5, required_radial_space=2.0
        ),
    ]

    fuzz_bounds = {
        "available_radial_space": (0.1, 10.0),
        "required_radial_space": (0.1, 10.0),
    }


def _reference_constraint_84(beta_total_vol_avg, beta_vol_avg_min):
    data = DataStructure()
    data.physics.beta_total_vol_avg = beta_total_vol_avg
    data.physics.beta_vol_avg_min = beta_vol_avg_min
    return _evaluate(84, data)


class TestConstraint84(Tier1Contract):
    """`constraint_equation_84` -> `constraint_84`."""

    audit_record = "core/solver/constraints.md"
    reference = _reference_constraint_84
    ported = constraint_84

    samples = [
        legacy_sample("feasible", beta_total_vol_avg=0.03, beta_vol_avg_min=0.01),
        legacy_sample("infeasible", beta_total_vol_avg=0.005, beta_vol_avg_min=0.01),
    ]

    fuzz_bounds = {
        "beta_total_vol_avg": (0.001, 0.1),
        "beta_vol_avg_min": (0.0, 0.05),
    }


def _reference_constraint_85(
    i_cp_lifetime, cplife, cplife_input, life_div_fpy, life_blkt_fpy, life_plant
):
    data = DataStructure()
    data.costs.i_cp_lifetime = i_cp_lifetime
    data.costs.cplife = cplife
    data.costs.cplife_input = cplife_input
    data.costs.life_div_fpy = life_div_fpy
    data.fwbs.life_blkt_fpy = life_blkt_fpy
    data.costs.life_plant = life_plant
    return _evaluate(85, data)


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

    samples = [
        legacy_sample("user-input", i_cp_lifetime=0, **_common),
        legacy_sample("divertor", i_cp_lifetime=1, **_common),
        legacy_sample("blanket", i_cp_lifetime=2, **_common),
        legacy_sample("plant", i_cp_lifetime=3, **_common),
    ]

    fuzz_bounds = {
        "cplife": (1.0, 60.0),
        "cplife_input": (1.0, 60.0),
        "life_div_fpy": (1.0, 60.0),
        "life_blkt_fpy": (1.0, 60.0),
        "life_plant": (1.0, 60.0),
    }
    fuzz_fixed = {"i_cp_lifetime": 1}


def _reference_constraint_86(dx_tf_turn_general, t_turn_tf_max):
    data = DataStructure()
    data.tfcoil.dx_tf_turn_general = dx_tf_turn_general
    data.tfcoil.t_turn_tf_max = t_turn_tf_max
    return _evaluate(86, data)


class TestConstraint86(Tier1Contract):
    """`constraint_equation_86` -> `constraint_86`."""

    audit_record = "core/solver/constraints.md"
    reference = _reference_constraint_86
    ported = constraint_86

    samples = [
        legacy_sample("feasible", dx_tf_turn_general=0.03, t_turn_tf_max=0.05),
        legacy_sample("infeasible", dx_tf_turn_general=0.07, t_turn_tf_max=0.05),
    ]

    fuzz_bounds = {
        "dx_tf_turn_general": (0.001, 0.2),
        "t_turn_tf_max": (0.001, 0.2),
    }


def _reference_constraint_87(p_cryo_plant_electric_mw, p_cryo_plant_electric_max_mw):
    data = DataStructure()
    data.heat_transport.p_cryo_plant_electric_mw = p_cryo_plant_electric_mw
    data.heat_transport.p_cryo_plant_electric_max_mw = p_cryo_plant_electric_max_mw
    return _evaluate(87, data)


class TestConstraint87(Tier1Contract):
    """`constraint_equation_87` -> `constraint_87`."""

    audit_record = "core/solver/constraints.md"
    reference = _reference_constraint_87
    ported = constraint_87

    samples = [
        legacy_sample(
            "feasible", p_cryo_plant_electric_mw=30.0, p_cryo_plant_electric_max_mw=50.0
        ),
        legacy_sample(
            "infeasible",
            p_cryo_plant_electric_mw=60.0,
            p_cryo_plant_electric_max_mw=50.0,
        ),
    ]
    fuzz_bounds = {
        "p_cryo_plant_electric_mw": (0.1, 100.0),
        "p_cryo_plant_electric_max_mw": (1.0, 200.0),
    }


def _reference_constraint_88(str_wp, str_wp_max):
    data = DataStructure()
    data.tfcoil.str_wp = str_wp
    data.tfcoil.str_wp_max = str_wp_max
    return _evaluate(88, data)


class TestConstraint88(Tier1Contract):
    """`constraint_equation_88` -> `constraint_88`."""

    audit_record = "core/solver/constraints.md"
    reference = _reference_constraint_88
    ported = constraint_88

    samples = [
        legacy_sample("feasible-positive", str_wp=0.003, str_wp_max=0.005),
        legacy_sample("feasible-negative", str_wp=-0.004, str_wp_max=0.005),
        legacy_sample("infeasible", str_wp=-0.006, str_wp_max=0.005),
    ]
    # `abs(str_wp)` is non-differentiable at str_wp == 0 -- keep fuzz bounds off zero,
    # same discipline as any other |.|-based constraint would need.
    fuzz_bounds = {
        "str_wp": (-0.02, -0.001),
        "str_wp_max": (0.001, 0.02),
    }


def _reference_constraint_89(copperaoh_m2, copperaoh_m2_max):
    data = DataStructure()
    data.rebco.copperaoh_m2 = copperaoh_m2
    data.rebco.copperaoh_m2_max = copperaoh_m2_max
    return _evaluate(89, data)


class TestConstraint89(Tier1Contract):
    """`constraint_equation_89` -> `constraint_89`."""

    audit_record = "core/solver/constraints.md"
    reference = _reference_constraint_89
    ported = constraint_89

    samples = [
        legacy_sample("feasible", copperaoh_m2=5.0e7, copperaoh_m2_max=1.0e8),
        legacy_sample("infeasible", copperaoh_m2=1.5e8, copperaoh_m2_max=1.0e8),
    ]
    fuzz_bounds = {
        "copperaoh_m2": (1.0e6, 2.0e8),
        "copperaoh_m2_max": (1.0e6, 2.0e8),
    }


def _reference_constraint_90(n_cycle, n_cycle_min, ibkt_life, bkt_life_csf, bktcycles):
    data = DataStructure()
    data.cs_fatigue.n_cycle = n_cycle
    data.cs_fatigue.n_cycle_min = n_cycle_min
    data.costs.ibkt_life = ibkt_life
    data.cs_fatigue.bkt_life_csf = bkt_life_csf
    data.costs.bktcycles = bktcycles
    return _evaluate(90, data)


class TestConstraint90(Tier1Contract):
    """`constraint_equation_90` -> `constraint_90`."""

    audit_record = "core/solver/constraints.md"
    reference = _reference_constraint_90
    ported = constraint_90

    static_argnames = ("ibkt_life", "bkt_life_csf")

    samples = [
        legacy_sample(
            "override-off-feasible",
            n_cycle=3.0e4,
            n_cycle_min=2.0e4,
            ibkt_life=0,
            bkt_life_csf=0.0,
            bktcycles=1.0e3,
        ),
        legacy_sample(
            "override-off-infeasible",
            n_cycle=1.0e4,
            n_cycle_min=2.0e4,
            ibkt_life=0,
            bkt_life_csf=0.0,
            bktcycles=1.0e3,
        ),
        # Override branch: n_cycle_min is replaced by bktcycles (1.0e3, much smaller
        # than the passed-in n_cycle_min of 2.0e4) -- this sample would fail if the
        # port ever stopped honouring the override, same discipline as constraint 24's
        # `stellarator-overrides-to-total` sample.
        legacy_sample(
            "override-on-uses-bktcycles",
            n_cycle=2.0e3,
            n_cycle_min=2.0e4,
            ibkt_life=1,
            bkt_life_csf=1.0,
            bktcycles=1.0e3,
        ),
        # Only one of the two switches on -- override must NOT fire.
        legacy_sample(
            "override-half-on-no-effect",
            n_cycle=2.0e3,
            n_cycle_min=2.0e4,
            ibkt_life=1,
            bkt_life_csf=0.0,
            bktcycles=1.0e3,
        ),
    ]
    fuzz_bounds = {
        "n_cycle": (1.0e3, 1.0e5),
        "n_cycle_min": (1.0e3, 1.0e5),
        "bktcycles": (1.0e2, 1.0e4),
    }
    fuzz_fixed = {"ibkt_life": 0, "bkt_life_csf": 0.0}


def _reference_constraint_91(
    i_plasma_ignited,
    p_hcd_primary_extra_heat_mw,
    powerht_constraint,
    powerscaling_constraint,
):
    """Call PROCESS's `constraint_equation_91` through the port's signature."""
    data = DataStructure()
    data.physics.i_plasma_ignited = i_plasma_ignited
    data.current_drive.p_hcd_primary_extra_heat_mw = p_hcd_primary_extra_heat_mw
    data.stellarator.powerht_constraint = powerht_constraint
    data.stellarator.powerscaling_constraint = powerscaling_constraint
    return _evaluate(91, data)


class TestConstraint91(Tier1Contract):
    """`constraint_equation_91` -> `constraint_91`."""

    audit_record = "core/solver/constraints.md"
    reference = _reference_constraint_91
    ported = constraint_91

    static_argnames = ("i_plasma_ignited",)

    samples = [
        legacy_sample(
            "non-ignited-feasible",
            i_plasma_ignited=int(PlasmaIgnitionModel.NON_IGNITED),
            p_hcd_primary_extra_heat_mw=5.0,
            powerht_constraint=120.0,
            powerscaling_constraint=100.0,
        ),
        legacy_sample(
            "ignited-feasible",
            i_plasma_ignited=int(PlasmaIgnitionModel.IGNITED),
            p_hcd_primary_extra_heat_mw=5.0,
            powerht_constraint=120.0,
            powerscaling_constraint=100.0,
        ),
        legacy_sample(
            "non-ignited-infeasible",
            i_plasma_ignited=int(PlasmaIgnitionModel.NON_IGNITED),
            p_hcd_primary_extra_heat_mw=1.0,
            powerht_constraint=10.0,
            powerscaling_constraint=100.0,
        ),
    ]

    fuzz_bounds = {
        "p_hcd_primary_extra_heat_mw": (0.0, 500.0),
        "powerht_constraint": (1.0, 1000.0),
        "powerscaling_constraint": (1.0, 1000.0),
    }
    fuzz_fixed = {"i_plasma_ignited": int(PlasmaIgnitionModel.NON_IGNITED)}


def _reference_constraint_92(
    f_plasma_fuel_deuterium, f_plasma_fuel_tritium, f_plasma_fuel_helium3
):
    data = DataStructure()
    data.physics.f_plasma_fuel_deuterium = f_plasma_fuel_deuterium
    data.physics.f_plasma_fuel_tritium = f_plasma_fuel_tritium
    data.physics.f_plasma_fuel_helium3 = f_plasma_fuel_helium3
    return _evaluate(92, data)


class TestConstraint92(Tier1Contract):
    """`constraint_equation_92` -> `constraint_92`."""

    audit_record = "core/solver/constraints.md"
    reference = _reference_constraint_92
    ported = constraint_92

    samples = [
        legacy_sample(
            "dt-only",
            f_plasma_fuel_deuterium=0.5,
            f_plasma_fuel_tritium=0.5,
            f_plasma_fuel_helium3=0.0,
        ),
        legacy_sample(
            "dt-plus-he3",
            f_plasma_fuel_deuterium=0.49,
            f_plasma_fuel_tritium=0.49,
            f_plasma_fuel_helium3=0.02,
        ),
        legacy_sample(
            "inconsistent",
            f_plasma_fuel_deuterium=0.5,
            f_plasma_fuel_tritium=0.4,
            f_plasma_fuel_helium3=0.0,
        ),
    ]
    fuzz_bounds = {
        "f_plasma_fuel_deuterium": (0.0, 1.0),
        "f_plasma_fuel_tritium": (0.0, 1.0),
        "f_plasma_fuel_helium3": (0.0, 1.0),
    }
