"""Harness cases for the ported coil quench protection (registry unit #14).

`max_dump_voltage`/`calculate_quench_protection_current_density` are already pure in
the source (no `data` argument) — their reference is the module-level function itself,
called directly, no `DataStructure` adapter needed. `calculate_quench_protection`'s
reference adapter derives `coilcurrent` from `c_tf_total`/`n_tf_coils` the same way the
port does internally (see `quench.md`'s "`coilcurrent` eliminated" note) before calling
the real PROCESS function, which still takes `coilcurrent` as an explicit argument.
"""

from functional_process._harness import Tier1Contract, legacy_sample
from functional_process.models.stellarator.coils.quench import (
    calculate_quench_protection,
    calculate_quench_protection_current_density,
    max_dump_voltage,
)
from process.core.model import DataStructure
from process.models.stellarator.coils.quench import (
    calculate_quench_protection as _process_calculate_quench_protection,
)
from process.models.stellarator.coils.quench import (
    calculate_quench_protection_current_density as _process_current_density,
)
from process.models.stellarator.coils.quench import max_dump_voltage as _process_max_dump_voltage


def _reference_quench_protection(
    rmajor,
    rminor,
    dr_fw_plasma_gap_inboard,
    dr_fw_inboard,
    dr_blkt_inboard,
    dr_shld_blkt_gap,
    dr_shld_inboard,
    dr_fw_plasma_gap_outboard,
    dr_fw_outboard,
    dr_blkt_outboard,
    dr_shld_outboard,
    b_plasma_toroidal_on_axis,
    c_tf_total,
    t_tf_superconductor_quench,
    dr_vv_inboard,
    dr_vv_outboard,
    t_tf_quench_detection,
    f_a_tf_turn_cable_copper,
    f_a_tf_turn_cable_space_extra_void,
    tftmp,
    a_tf_turn_cable_space_no_void,
    dx_tf_turn_general,
    a_tf_wp_conductor,
    e_tf_magnetic_stored_total_gj,
    n_tf_coils,
    c_tf_turn,
):
    """Call PROCESS's `calculate_quench_protection` through the port's signature."""
    data = DataStructure()
    data.physics.rmajor = rmajor
    data.physics.rminor = rminor
    data.build.dr_fw_plasma_gap_inboard = dr_fw_plasma_gap_inboard
    data.build.dr_fw_inboard = dr_fw_inboard
    data.build.dr_blkt_inboard = dr_blkt_inboard
    data.build.dr_shld_blkt_gap = dr_shld_blkt_gap
    data.build.dr_shld_inboard = dr_shld_inboard
    data.build.dr_fw_plasma_gap_outboard = dr_fw_plasma_gap_outboard
    data.build.dr_fw_outboard = dr_fw_outboard
    data.build.dr_blkt_outboard = dr_blkt_outboard
    data.build.dr_shld_outboard = dr_shld_outboard
    data.physics.b_plasma_toroidal_on_axis = b_plasma_toroidal_on_axis
    data.tfcoil.c_tf_total = c_tf_total
    data.tfcoil.t_tf_superconductor_quench = t_tf_superconductor_quench
    data.build.dr_vv_inboard = dr_vv_inboard
    data.build.dr_vv_outboard = dr_vv_outboard
    data.tfcoil.t_tf_quench_detection = t_tf_quench_detection
    data.tfcoil.f_a_tf_turn_cable_copper = f_a_tf_turn_cable_copper
    data.tfcoil.f_a_tf_turn_cable_space_extra_void = f_a_tf_turn_cable_space_extra_void
    data.tfcoil.tftmp = tftmp
    data.tfcoil.a_tf_turn_cable_space_no_void = a_tf_turn_cable_space_no_void
    data.tfcoil.dx_tf_turn_general = dx_tf_turn_general
    data.tfcoil.a_tf_wp_conductor = a_tf_wp_conductor
    data.tfcoil.e_tf_magnetic_stored_total_gj = e_tf_magnetic_stored_total_gj
    data.tfcoil.n_tf_coils = n_tf_coils
    data.tfcoil.c_tf_turn = c_tf_turn

    coilcurrent = c_tf_total / (n_tf_coils * 1.0e6)
    f_vv_actual = _process_calculate_quench_protection(coilcurrent, data)

    return (
        f_vv_actual,
        data.superconducting_tfcoil.vv_stress_quench,
        data.tfcoil.j_tf_wp_quench_heat_max,
        data.rebco.coppera_m2,
        data.tfcoil.v_tf_coil_dump_quench_kv,
    )


class TestQuenchProtection(Tier1Contract):
    """`calculate_quench_protection` (PROCESS, `coilcurrent` dropped) -> the port."""

    audit_record = "models/stellarator/coils/quench.md"
    reference = _reference_quench_protection
    ported = calculate_quench_protection

    # Realistic helias5b-scale point, hand-assembled (no direct PROCESS unit test
    # exercises the full chain) and confirmed to agree with the real PROCESS function
    # exactly before this file was written (see quench.md's verification note).
    samples = [
        legacy_sample(
            "quench-helias5b-scale",
            rmajor=22.0,
            rminor=1.78,
            dr_fw_plasma_gap_inboard=0.02,
            dr_fw_inboard=0.018,
            dr_blkt_inboard=0.83,
            dr_shld_blkt_gap=0.05,
            dr_shld_inboard=0.2,
            dr_fw_plasma_gap_outboard=0.02,
            dr_fw_outboard=0.018,
            dr_blkt_outboard=1.08,
            dr_shld_outboard=0.2,
            b_plasma_toroidal_on_axis=5.5,
            c_tf_total=3.2e8,
            t_tf_superconductor_quench=15.0,
            dr_vv_inboard=0.3,
            dr_vv_outboard=0.3,
            t_tf_quench_detection=3.0,
            f_a_tf_turn_cable_copper=0.69,
            f_a_tf_turn_cable_space_extra_void=0.3,
            tftmp=4.2,
            a_tf_turn_cable_space_no_void=0.0022,
            dx_tf_turn_general=0.056,
            a_tf_wp_conductor=0.5,
            e_tf_magnetic_stored_total_gj=132.5,
            n_tf_coils=50,
            c_tf_turn=65000.0,
        ),
    ]

    fuzz_bounds = {
        "rmajor": (5.0, 30.0),
        "rminor": (0.5, 5.0),
        "dr_fw_plasma_gap_inboard": (0.005, 0.1),
        "dr_fw_inboard": (0.005, 0.1),
        "dr_blkt_inboard": (0.1, 2.0),
        "dr_shld_blkt_gap": (0.005, 0.2),
        "dr_shld_inboard": (0.05, 1.0),
        "dr_fw_plasma_gap_outboard": (0.005, 0.1),
        "dr_fw_outboard": (0.005, 0.1),
        "dr_blkt_outboard": (0.1, 2.0),
        "dr_shld_outboard": (0.05, 1.0),
        "b_plasma_toroidal_on_axis": (1.0, 20.0),
        "c_tf_total": (1.0e7, 1.0e9),
        "t_tf_superconductor_quench": (1.0, 60.0),
        "dr_vv_inboard": (0.05, 1.0),
        "dr_vv_outboard": (0.05, 1.0),
        "t_tf_quench_detection": (0.1, 20.0),
        "f_a_tf_turn_cable_copper": (0.1, 0.95),
        "f_a_tf_turn_cable_space_extra_void": (0.01, 0.9),
        "tftmp": (1.0, 20.0),
        "a_tf_turn_cable_space_no_void": (1.0e-4, 0.1),
        "dx_tf_turn_general": (0.005, 0.5),
        "a_tf_wp_conductor": (0.01, 5.0),
        "e_tf_magnetic_stored_total_gj": (1.0, 1.0e4),
        "n_tf_coils": (1.0, 100.0),
        "c_tf_turn": (100.0, 1.0e6),
    }


class TestMaxDumpVoltage(Tier1Contract):
    """`max_dump_voltage`, already pure in the source -- direct port."""

    audit_record = "models/stellarator/coils/quench.md"
    reference = staticmethod(_process_max_dump_voltage)
    ported = max_dump_voltage

    samples = [
        # tests/unit/models/stellarator/test_stellarator.py::test_u_max_protect_v
        legacy_sample(
            "u_max_protect_v",
            tf_energy_stored=2651198129.2530489,
            t_dump=10,
            current=122620.32643505408,
        ),
    ]

    fuzz_bounds = {
        "tf_energy_stored": (1.0e6, 1.0e11),
        "t_dump": (0.5, 100.0),
        "current": (1.0e3, 1.0e7),
    }


class TestQuenchProtectionCurrentDensity(Tier1Contract):
    """`calculate_quench_protection_current_density`, already pure -- direct port."""

    audit_record = "models/stellarator/coils/quench.md"
    reference = staticmethod(_process_current_density)
    ported = calculate_quench_protection_current_density

    samples = [
        # tests/unit/models/stellarator/test_stellarator.py::test_j_max_protect_am2
        legacy_sample(
            "j_max_protect_am2",
            tau_quench=10,
            t_detect=0,
            f_cu=0.69000000000000017,
            f_cond=0.69999999999999996,
            temp=4.2000000000000002,
            a_cable=0.0022141440000000008,
            a_turn=0.0031360000000000008,
        ),
    ]

    fuzz_bounds = {
        "tau_quench": (1.0, 60.0),
        "t_detect": (0.0, 20.0),
        "f_cu": (0.1, 0.95),
        "f_cond": (0.1, 0.95),
        "temp": (4.0, 124.0),
        "a_cable": (1.0e-4, 0.1),
        "a_turn": (1.0e-4, 0.1),
    }
