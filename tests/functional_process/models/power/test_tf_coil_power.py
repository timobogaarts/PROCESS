"""Harness cases for `functional_process/models/power/tf_coil_power.py`.

Audit record: `functional_process/_audit/units/models/power/tf_coil_power.md`. No
legacy points exist for either function in `tests/unit/models/test_power.py`
(`tfpwr`/`tfpwcall`/
`tfcpwr` have no automatically-generated unit test there, unlike `cryo`/`acpow`/
`plant_electric_production` -- see `thermal_cryo.md`/`electric_production.md` for
those) -- fuzz-only, same situation `build.md` documented for
`st_build`.
"""

from functional_process._harness import Tier1Contract, fuzz_samples
from functional_process.models.power.tf_coil_power import (
    calculate_tf_power_resistive,
    calculate_tf_power_superconducting,
)
from process.core.model import DataStructure
from process.models.power import Power


def _reference_tf_power_resistive(
    c_tf_turn,
    j_tf_bus,
    rho_tf_bus,
    len_tf_bus,
    n_tf_coils,
    res_tf_leg,
    p_cp_resistive,
    c_tf_total,
    p_tf_joints_resistive,
    p_tf_leg_resistive,
    etatf,
):
    """Call PROCESS's `Power.tfpwr` (resistive branch) through the port's signature."""
    data = DataStructure()
    data.tfcoil.i_tf_sup = 0
    data.tfcoil.c_tf_turn = c_tf_turn
    data.tfcoil.j_tf_bus = j_tf_bus
    data.tfcoil.rho_tf_bus = rho_tf_bus
    data.tfcoil.len_tf_bus = len_tf_bus
    data.tfcoil.n_tf_coils = n_tf_coils
    data.tfcoil.res_tf_leg = res_tf_leg
    data.tfcoil.p_cp_resistive = p_cp_resistive
    data.tfcoil.c_tf_total = c_tf_total
    data.tfcoil.p_tf_joints_resistive = p_tf_joints_resistive
    data.tfcoil.p_tf_leg_resistive = p_tf_leg_resistive
    data.heat_transport.etatf = etatf

    p = Power()
    p.data = data
    p.tfpwr(output=False)

    return (
        data.tfcoil.m_tf_bus,
        data.tfcoil.vtfkv,
        data.tfcoil.p_cp_resistive_mw,
        data.tfcoil.p_tf_leg_resistive_mw,
        data.tfcoil.p_tf_joints_resistive_mw,
        data.tfcoil.tfcmw,
        data.heat_transport.p_tf_electric_supplies_mw,
    )


def _reference_tf_power_superconducting(
    c_tf_turn,
    e_tf_magnetic_stored_total_gj,
    n_tf_coils,
    rmajor,
    v_tf_coil_dump_quench_kv,
    res_tf_leg,
    rho_tf_bus,
    etatf,
):
    """Call PROCESS's `Power.tfpwr` (superconducting branch, via `tfpwcall`)."""
    data = DataStructure()
    data.tfcoil.i_tf_sup = 1
    data.tfcoil.c_tf_turn = c_tf_turn
    data.tfcoil.e_tf_magnetic_stored_total_gj = e_tf_magnetic_stored_total_gj
    data.tfcoil.n_tf_coils = n_tf_coils
    data.physics.rmajor = rmajor
    data.tfcoil.v_tf_coil_dump_quench_kv = v_tf_coil_dump_quench_kv
    data.tfcoil.res_tf_leg = res_tf_leg
    data.tfcoil.rho_tf_bus = rho_tf_bus
    data.heat_transport.etatf = etatf

    p = Power()
    p.data = data
    p.tfpwr(output=False)

    return (
        data.tfcoil.tfckw,
        data.tfcoil.len_tf_bus,
        data.tfcoil.drarea,
        data.buildings.tfcbv,
        data.heat_transport.p_tf_electric_supplies_mw,
    )


class TestTfPowerResistive(Tier1Contract):
    """`Power.tfpwr`'s `i_tf_sup != 1` branch -> `calculate_tf_power_resistive`."""

    audit_record = "models/power/tf_coil_power.md"
    reference = _reference_tf_power_resistive
    ported = calculate_tf_power_resistive

    fuzz_bounds = {
        "c_tf_turn": (1.0e3, 1.5e5),
        "j_tf_bus": (1.0e5, 1.0e7),
        "rho_tf_bus": (1.0e-8, 5.0e-8),
        "len_tf_bus": (10.0, 2000.0),
        "n_tf_coils": (10.0, 24.0),
        "res_tf_leg": (1.0e-7, 1.0e-5),
        "p_cp_resistive": (1.0e5, 5.0e7),
        "c_tf_total": (1.0e6, 2.0e8),
        "p_tf_joints_resistive": (0.0, 1.0e6),
        "p_tf_leg_resistive": (0.0, 1.0e7),
        "etatf": (0.6, 1.0),
    }
    samples = fuzz_samples(fuzz_bounds, count=40, seed=20260818)


def _tf_power_superconducting_samples():
    """Fuzz points within `res_tf_leg`'s "resistive-leg" sub-case.

    `calculate_tf_power_superconducting`'s only real branch (inside PROCESS's own
    `tfcpwr`) is `res_tf_leg == 0.0` exactly (see the function's docstring) --
    deliberately *not* sampled here. Verified by hand (not through this harness,
    see `tf_coil_power.md`'s open questions) that the port agrees with
    PROCESS to float64 round-off at `res_tf_leg = 0.0` -- but `jax.jacfwd` produces a
    `NaN` there for *every* differentiated argument, not just `res_tf_leg` itself: the
    `jnp.sqrt(n_tf_coils * res_tf_leg * 1000.0)` term's JVP rule divides by
    `2 * sqrt(primal)`, and at `res_tf_leg = 0.0` that is `0 * inf = NaN` regardless of
    which input's tangent is flowing through it. This is not a `jnp.where`-hides-a-bug
    artifact to guard against -- the source formula's true analytic derivative really
    is unbounded at that point (a genuine square-root singularity), so no guard would
    make it a different, still-correct number. `res_tf_leg = 0.0` is also not the
    regime this field is used in practice (PROCESS's own docs and this port's fuzz
    range treat it as a small positive leg resistance) -- flagged as a JAX-difficulty,
    not exercised here.
    """
    bounds = {
        "c_tf_turn": (1.0e3, 1.5e5),
        "e_tf_magnetic_stored_total_gj": (0.5, 30.0),
        "n_tf_coils": (10.0, 24.0),
        "rmajor": (3.0, 20.0),
        "v_tf_coil_dump_quench_kv": (1.0, 20.0),
        "res_tf_leg": (1.0e-7, 1.0e-5),
        "rho_tf_bus": (1.0e-8, 5.0e-8),
        "etatf": (0.6, 1.0),
    }
    return fuzz_samples(bounds, count=40, seed=20260818)


class TestTfPowerSuperconducting(Tier1Contract):
    """`Power.tfpwr`'s `i_tf_sup == 1` branch -> `calculate_tf_power_superconducting`."""

    audit_record = "models/power/tf_coil_power.md"
    reference = _reference_tf_power_superconducting
    ported = calculate_tf_power_superconducting
    samples = _tf_power_superconducting_samples()
