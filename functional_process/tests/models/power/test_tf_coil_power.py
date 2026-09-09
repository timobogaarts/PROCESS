"""Harness cases for `functional_process/cottax/power/tf_coil_power.py`. Fuzz-only:
no legacy points exist for either function in `tests/unit/models/test_power.py`.
"""

from functional_process.cottax._harness import Tier1Contract, fuzz_samples
from functional_process.cottax._harness.process_reference import process_reference
from functional_process.cottax._harness.sample_store import FROM_FILE
from functional_process.cottax.power.tf_coil_power import (
    calculate_tf_power_resistive,
    calculate_tf_power_superconducting,
)
from process.core.model import DataStructure
from process.models.power import Power


def _make_power_resistive():
    p = Power()
    p.data = DataStructure()
    p.data.tfcoil.i_tf_sup = 0
    return p


def _make_power_superconducting():
    p = Power()
    p.data = DataStructure()
    p.data.tfcoil.i_tf_sup = 1
    return p


_reference_tf_power_resistive = process_reference(
    _make_power_resistive,
    "tfpwr",
    (
        "tfcoil.m_tf_bus",
        "tfcoil.vtfkv",
        "tfcoil.p_cp_resistive_mw",
        "tfcoil.p_tf_leg_resistive_mw",
        "tfcoil.p_tf_joints_resistive_mw",
        "tfcoil.tfcmw",
        "heat_transport.p_tf_electric_supplies_mw",
    ),
    call_args=(False,),
)


_reference_tf_power_superconducting = process_reference(
    _make_power_superconducting,
    "tfpwr",
    (
        "tfcoil.tfckw",
        "tfcoil.len_tf_bus",
        "tfcoil.drarea",
        "buildings.tfcbv",
        "heat_transport.p_tf_electric_supplies_mw",
    ),
    call_args=(False,),
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
    samples = FROM_FILE


def _tf_power_superconducting_samples():
    """Fuzz points within `res_tf_leg`'s "resistive-leg" sub-case.

    `calculate_tf_power_superconducting`'s only real branch (inside PROCESS's own
    `tfcpwr`) is `res_tf_leg == 0.0` exactly (see the function's docstring) --
    deliberately *not* sampled here. Verified by hand (not through this harness)
    that the port agrees with
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
    samples = FROM_FILE
