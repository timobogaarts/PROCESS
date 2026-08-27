"""Harness cases for `functional_process/models/pfcoil/masses.py`, and for every inline
block of `PFCoil.pfcoil()` the package ports.

Audit record: `functional_process/_audit/units/models/pfcoil/masses.md`.

**One contract, whose reference is `PFCoil.pfcoil()` itself.** Most of what this package
ports is not a PROCESS function: `pfcoil()` is a single 1023-line routine, and the
coil-position flattening, the plasma-initiation and equilibrium current blocks, the CS
flux swing, the time-point currents, the winding-pack sizing and the mass summations are
all inline stretches of it with no callable of their own. Testing each against a
hand-written "reference" would be testing a copy of the port against the port. So the
oracle here is PROCESS's own routine, driven from a cold `DataStructure` seeded with
exactly the inputs the port declares, and the ported side is those blocks composed in
`pfcoil()`'s own order -- `_ported_pf_coil_chain` below, which introduces no arithmetic
of its own and calls only the package's public functions.

That makes this contract cover, in one place: `geometry.calculate_cs_geometry`,
`geometry.calculate_pf_coil_group_positions`, `geometry.calculate_pf_coil_positions`,
`currents.calculate_plasma_initiation_currents`,
`currents.calculate_equilibrium_currents`, `currents.calculate_cs_flux_swing`,
`currents.calculate_time_point_currents`, `fields.calculate_coil_current_waveform`,
`fields.calculate_pf_coil_peak_fields`, `masses.calculate_pf_coil_sizes` and
`masses.calculate_pf_coil_masses`. The four units that *do* have a separable PROCESS
callable keep their own narrower contracts in the sibling test modules, so a failure
there localises what a failure here only detects.

**The two loop-carried inputs are arguments, on both sides.** `pfcoil()` reads
`.pf_coil.n_pf_coil_turns` and `.pf_coil.ind_pf_cs_plasma_mutual` before writing them --
the first from its own previous pass, the second from `induct()`, which this port does
not cover. They are therefore ordinary inputs of the composed function *and* seeded
fields of the reference's `DataStructure`, with `first_call = False` so PROCESS does not
overwrite them with its bootstrap values (`process/models/pfcoil.py:605-608`). This is
the cycle `currents.py`'s module docstring describes, cut in the same place on both
sides so the comparison is of one pass against one pass.

**Fuzzing is +-10% around the converged point, not over an invented range.** `pfcoil()`
raises rather than returning non-finite on a bad topology, `superconpf` inside `ohcalc()`
runs a Newton solve that need not converge, and the CS stress fits go complex outside
their domain. None of that is what this contract is measuring, and all of it is
reachable from a wide draw. `+-10%` keeps every draw a plausible machine while still
moving every declared input.
"""

import numpy as np

from functional_process._harness import Tier1Contract, Tolerance, legacy_sample
from functional_process.models.pfcoil import (
    N_COILS_IN_GROUP,
    N_CS_FILAMENTS,
    N_CS_PF_COILS,
    N_PF_COILS,
    N_PF_GROUPS,
    NGC2,
    PLASMA_INDEX,
)
from functional_process.models.pfcoil.currents import (
    calculate_cs_flux_swing,
    calculate_equilibrium_currents,
    calculate_plasma_initiation_currents,
    calculate_time_point_currents,
)
from functional_process.models.pfcoil.fields import (
    calculate_coil_current_waveform,
    calculate_pf_coil_peak_fields,
)
from functional_process.models.pfcoil.geometry import (
    calculate_cs_geometry,
    calculate_pf_coil_group_positions,
    calculate_pf_coil_positions,
)
from functional_process.models.pfcoil.masses import (
    calculate_pf_coil_masses,
    calculate_pf_coil_sizes,
)
from functional_process.models.pfcoil.volt_seconds import (
    calculate_pf_coil_turn_currents,
)
from process.core.exceptions import ProcessValueError
from process.core.model import DataStructure
from process.models.cs_fatigue import CsFatigue
from process.models.pfcoil import CSCoil, PFCoil

I_PF_LOCATION = np.array([2, 2, 3, 3, 0, 0, 0, 0, 0, 0])
"""`.pf_coil.i_pf_location` on the reference run -- two groups above the TF coil, two
outside it."""


def _ported_pf_coil_chain(
    z_tf_inside_half,
    f_z_cs_tf_internal,
    dr_cs,
    dr_cs_bore,
    z_tf_top,
    dz_tf_upper_lower_midplane,
    rpf2,
    zref,
    r_tf_outboard_out,
    dr_pf_tf_outboard_out_offset,
    rmajor,
    rminor,
    kappa,
    triang,
    aspect,
    plasma_current,
    beta_poloidal_vol_avg,
    ind_plasma_internal_norm,
    vs_plasma_ramp_required,
    alfapf,
    j_cs_flat_top_end,
    f_j_cs_start_pulse_end_flat_top,
    ind_pf_cs_plasma_mutual_column,
    n_pf_coil_turns_previous,
    j_pf_coil_wp_peak,
    c_pf_coil_turn_peak_input,
    pf_current_safety_factor,
    f_a_pf_coil_void,
    sigpfcf,
    sigpfcalw,
    den_steel,
    den_pf_conductor,
    den_cs_conductor,
    f_a_cs_turn_steel,
    f_a_cs_void,
):
    """The package's public functions composed in `PFCoil.pfcoil()`'s own order.

    Not part of the port: nothing here computes anything the port does not already
    compute, and no node binds this. It exists so that the inline blocks of `pfcoil()`
    have PROCESS's `pfcoil()` as an oracle instead of a re-implementation.
    """
    (
        z_cs_upper,
        z_cs_lower,
        _r_cs_coil_middle,
        r_cs_middle,
        _z_cs_middle,
        r_cs_outer,
        r_cs_inner,
        a_cs_poloidal,
        _a_cs_toroidal,
        dz_cs_full,
        _dr_cs_full,
    ) = calculate_cs_geometry(
        z_tf_inside_half=z_tf_inside_half,
        f_z_cs_tf_internal=f_z_cs_tf_internal,
        dr_cs=dr_cs,
        dr_cs_bore=dr_cs_bore,
    )

    r_pf_outside_tf_midplane = r_tf_outboard_out + dr_pf_tf_outboard_out_offset
    r_group, z_group = calculate_pf_coil_group_positions(
        rmajor=rmajor,
        rminor=rminor,
        triang=triang,
        rpf2=rpf2,
        z_tf_top=z_tf_top,
        dz_tf_upper_lower_midplane=dz_tf_upper_lower_midplane,
        zref=zref,
        r_pf_outside_tf_midplane=r_pf_outside_tf_midplane,
    )
    r_pf_coil_middle, z_pf_coil_middle = calculate_pf_coil_positions(
        r_pf_coil_middle_group_array=r_group,
        z_pf_coil_middle_group_array=z_group,
        r_cs_middle=r_cs_middle,
    )

    j_cs_pulse_start = j_cs_flat_top_end * f_j_cs_start_pulse_end_flat_top

    ssq0, ccl0 = calculate_plasma_initiation_currents(
        rmajor=rmajor,
        rminor=rminor,
        r_pf_coil_middle_group_array=r_group,
        z_pf_coil_middle_group_array=z_group,
        r_cs_middle=r_cs_middle,
        dz_cs_full=dz_cs_full,
        a_cs_poloidal=a_cs_poloidal,
        j_cs_flat_top_end=j_cs_flat_top_end,
        f_j_cs_start_pulse_end_flat_top=f_j_cs_start_pulse_end_flat_top,
        alfapf=alfapf,
    )
    ccls, b_plasma_vertical_required = calculate_equilibrium_currents(
        rmajor=rmajor,
        rminor=rminor,
        kappa=kappa,
        aspect=aspect,
        plasma_current=plasma_current,
        beta_poloidal_vol_avg=beta_poloidal_vol_avg,
        ind_plasma_internal_norm=ind_plasma_internal_norm,
        r_pf_coil_middle_group_array=r_group,
        z_pf_coil_middle_group_array=z_group,
        alfapf=alfapf,
    )
    f_j_cs_start_end_flat_top = calculate_cs_flux_swing(
        ccls=ccls[:N_PF_GROUPS],
        ind_pf_cs_plasma_mutual_column=ind_pf_cs_plasma_mutual_column,
        n_pf_coil_turns=n_pf_coil_turns_previous,
        vs_plasma_ramp_required=vs_plasma_ramp_required,
        dr_cs_bore=dr_cs_bore,
        dr_cs=dr_cs,
        dz_cs_full=dz_cs_full,
        a_cs_poloidal=a_cs_poloidal,
        j_cs_flat_top_end=j_cs_flat_top_end,
        f_j_cs_start_pulse_end_flat_top=f_j_cs_start_pulse_end_flat_top,
    )
    c_start, c_flat, c_end = calculate_time_point_currents(
        ccl0=ccl0,
        ccls=ccls,
        a_cs_poloidal=a_cs_poloidal,
        j_cs_flat_top_end=j_cs_flat_top_end,
        f_j_cs_start_pulse_end_flat_top=f_j_cs_start_pulse_end_flat_top,
        f_j_cs_start_end_flat_top=f_j_cs_start_end_flat_top,
    )
    c_peak, f_c_peak_time = calculate_coil_current_waveform(c_start, c_flat, c_end)

    # `pfcoil()`'s own tail (`pfcoil.py:1082-1111`) -- added to the chain 2026-08-27
    # with `volt_seconds.py::PFCoilTurnCurrents`, whose only PROCESS oracle this is.
    c_pf_coil_turn = calculate_pf_coil_turn_currents(
        f_c_pf_cs_peak_time_array=f_c_peak_time,
        c_pf_coil_turn_peak_input=c_pf_coil_turn_peak_input,
        c_pf_cs_coils_peak_ma=c_peak,
        plasma_current=plasma_current,
    )

    (
        n_pf_coil_turns,
        r_pf_coil_inner,
        r_pf_coil_outer,
        z_pf_coil_upper,
        z_pf_coil_lower,
        r_pf_coil_outer_max,
    ) = calculate_pf_coil_sizes(
        c_pf_cs_coils_peak_ma=c_peak,
        j_pf_coil_wp_peak=j_pf_coil_wp_peak,
        c_pf_coil_turn_peak_input=c_pf_coil_turn_peak_input,
        r_pf_coil_middle=r_pf_coil_middle,
        z_pf_coil_middle=z_pf_coil_middle,
        pf_current_safety_factor=pf_current_safety_factor,
        r_cs_inner=r_cs_inner,
        r_cs_outer=r_cs_outer,
        z_cs_upper=z_cs_upper,
        z_cs_lower=z_cs_lower,
        rmajor=rmajor,
        rminor=rminor,
        kappa=kappa,
    )

    b_pf_coil_peak, bpf2 = calculate_pf_coil_peak_fields(
        c_pf_cs_coil_pulse_start_ma=c_start,
        c_pf_cs_coil_flat_top_ma=c_flat,
        c_pf_cs_coil_pulse_end_ma=c_end,
        r_pf_coil_middle=r_pf_coil_middle,
        z_pf_coil_middle=z_pf_coil_middle,
        r_pf_coil_inner=r_pf_coil_inner[:N_CS_PF_COILS],
        r_pf_coil_outer=r_pf_coil_outer[:N_CS_PF_COILS],
        z_pf_coil_upper=z_pf_coil_upper[:N_CS_PF_COILS],
        z_pf_coil_lower=z_pf_coil_lower[:N_CS_PF_COILS],
        r_pf_coil_middle_group_array=r_group,
        z_pf_coil_middle_group_array=z_group,
        r_cs_middle=r_cs_middle,
        dz_cs_full=dz_cs_full,
        a_cs_poloidal=a_cs_poloidal,
        j_cs_pulse_start=j_cs_pulse_start,
        j_cs_flat_top_end=j_cs_flat_top_end,
        rmajor=rmajor,
        plasma_current=plasma_current,
    )

    (
        m_pf_coil_conductor,
        m_pf_coil_structure,
        pfcaseth,
        m_pf_coil_conductor_total,
        m_pf_coil_structure_total,
        m_pf_coil_max,
        ricpf,
        a_cs_steel_poloidal,
        a_cs_cable_space,
    ) = calculate_pf_coil_masses(
        c_pf_cs_coils_peak_ma=c_peak,
        j_pf_coil_wp_peak=j_pf_coil_wp_peak,
        n_pf_coil_turns=n_pf_coil_turns[:N_CS_PF_COILS],
        r_pf_coil_middle=r_pf_coil_middle,
        r_pf_coil_inner=r_pf_coil_inner[:N_CS_PF_COILS],
        r_pf_coil_outer=r_pf_coil_outer[:N_CS_PF_COILS],
        z_pf_coil_upper=z_pf_coil_upper[:N_CS_PF_COILS],
        z_pf_coil_lower=z_pf_coil_lower[:N_CS_PF_COILS],
        b_pf_coil_peak=b_pf_coil_peak,
        bpf2=bpf2,
        f_a_pf_coil_void=f_a_pf_coil_void,
        pf_current_safety_factor=pf_current_safety_factor,
        sigpfcf=sigpfcf,
        sigpfcalw=sigpfcalw,
        den_steel=den_steel,
        den_pf_conductor=den_pf_conductor,
        den_cs_conductor=den_cs_conductor,
        a_cs_poloidal=a_cs_poloidal,
        f_a_cs_turn_steel=f_a_cs_turn_steel,
        f_a_cs_void=f_a_cs_void,
    )

    return (
        m_pf_coil_conductor_total,
        m_pf_coil_structure_total,
        r_pf_coil_outer,
        m_pf_coil_conductor,
        m_pf_coil_structure,
        pfcaseth,
        m_pf_coil_max,
        ricpf,
        a_cs_steel_poloidal,
        a_cs_cable_space,
        r_pf_coil_outer_max,
        n_pf_coil_turns,
        r_pf_coil_inner,
        z_pf_coil_upper,
        z_pf_coil_lower,
        r_pf_coil_middle,
        z_pf_coil_middle,
        ccl0,
        ccls,
        ssq0,
        b_plasma_vertical_required,
        f_j_cs_start_end_flat_top,
        c_start,
        c_flat,
        c_end,
        c_peak,
        b_pf_coil_peak,
        bpf2,
        r_pf_outside_tf_midplane,
        a_cs_poloidal,
        dz_cs_full,
        c_pf_coil_turn[: PLASMA_INDEX + 1],
    )


def _reference_pf_coil_chain(
    z_tf_inside_half,
    f_z_cs_tf_internal,
    dr_cs,
    dr_cs_bore,
    z_tf_top,
    dz_tf_upper_lower_midplane,
    rpf2,
    zref,
    r_tf_outboard_out,
    dr_pf_tf_outboard_out_offset,
    rmajor,
    rminor,
    kappa,
    triang,
    aspect,
    plasma_current,
    beta_poloidal_vol_avg,
    ind_plasma_internal_norm,
    vs_plasma_ramp_required,
    alfapf,
    j_cs_flat_top_end,
    f_j_cs_start_pulse_end_flat_top,
    ind_pf_cs_plasma_mutual_column,
    n_pf_coil_turns_previous,
    j_pf_coil_wp_peak,
    c_pf_coil_turn_peak_input,
    pf_current_safety_factor,
    f_a_pf_coil_void,
    sigpfcf,
    sigpfcalw,
    den_steel,
    den_pf_conductor,
    den_cs_conductor,
    f_a_cs_turn_steel,
    f_a_cs_void,
):
    """`PFCoil.pfcoil()` on a cold `DataStructure` seeded with exactly these inputs.

    Everything set below is either one of the port's declared inputs, one of the
    topology constants the port bakes in (`__init__`'s table), or a switch this occupant
    answers. Every other field keeps its `DataStructure` default -- which is the point:
    if the port's declared read set were short of an input `pfcoil()` actually uses, that
    field would sit at its default here and the two sides would disagree.

    `.physics.f_c_plasma_inductive` is left at its default `0.0` so `ohcalc()` skips the
    CS fatigue calculation (`pfcoil.py:3488-3498`); that chain is UNPORTED, needs six
    more `cs_fatigue` inputs, and feeds nothing this contract compares.
    """
    data = DataStructure()
    build, p, physics = data.build, data.pf_coil, data.physics

    build.iohcl = 1
    build.z_tf_inside_half = z_tf_inside_half
    build.dr_cs = dr_cs
    build.dr_cs_bore = dr_cs_bore
    build.z_tf_top = z_tf_top
    build.dz_tf_upper_lower_midplane = dz_tf_upper_lower_midplane

    p.n_pf_coil_groups = N_PF_GROUPS
    p.n_pf_coils_in_group = np.array(
        [*N_COILS_IN_GROUP, 0, 0, 0, 0, 0, 0, 0, 0], dtype=int
    )
    p.i_pf_location = I_PF_LOCATION.copy()
    p.i_pf_conductor = 0
    p.i_pf_current = 1
    p.i_r_pf_outside_tf_placement = 0
    p.i_pf_superconductor = 3
    p.i_cs_superconductor = 1
    p.n_cs_current_filaments = N_CS_FILAMENTS
    p.first_call = False

    p.f_z_cs_tf_internal = f_z_cs_tf_internal
    p.rpf2 = rpf2
    p.zref = np.concatenate([np.asarray(zref, dtype=float), np.ones(6)])
    p.dr_pf_tf_outboard_out_offset = dr_pf_tf_outboard_out_offset
    p.alfapf = alfapf
    p.j_cs_flat_top_end = j_cs_flat_top_end
    p.f_j_cs_start_pulse_end_flat_top = f_j_cs_start_pulse_end_flat_top
    p.pf_current_safety_factor = pf_current_safety_factor
    p.sigpfcf = sigpfcf
    p.sigpfcalw = sigpfcalw
    p.f_a_cs_turn_steel = f_a_cs_turn_steel
    p.f_a_cs_void = f_a_cs_void

    p.j_pf_coil_wp_peak = np.zeros(NGC2)
    p.j_pf_coil_wp_peak[:N_CS_PF_COILS] = j_pf_coil_wp_peak
    p.c_pf_coil_turn_peak_input = np.zeros(NGC2)
    p.c_pf_coil_turn_peak_input[:N_CS_PF_COILS] = c_pf_coil_turn_peak_input
    p.f_a_pf_coil_void = np.full(NGC2, 0.3)
    p.f_a_pf_coil_void[:N_PF_COILS] = f_a_pf_coil_void

    p.ind_pf_cs_plasma_mutual = np.zeros((NGC2, NGC2))
    p.ind_pf_cs_plasma_mutual[:N_PF_COILS, PLASMA_INDEX] = ind_pf_cs_plasma_mutual_column
    p.n_pf_coil_turns = np.full(NGC2, 100.0)
    p.n_pf_coil_turns[:N_PF_COILS] = n_pf_coil_turns_previous

    data.superconducting_tfcoil.r_tf_outboard_out = r_tf_outboard_out
    data.fwbs.den_steel = den_steel
    data.tfcoil.dcond = np.asarray(data.tfcoil.dcond, dtype=float).copy()
    data.tfcoil.dcond[2] = den_pf_conductor
    data.tfcoil.dcond[0] = den_cs_conductor

    physics.rmajor = rmajor
    physics.rminor = rminor
    physics.kappa = kappa
    physics.triang = triang
    physics.aspect = aspect
    physics.plasma_current = plasma_current
    physics.beta_poloidal_vol_avg = beta_poloidal_vol_avg
    physics.ind_plasma_internal_norm = ind_plasma_internal_norm
    physics.vs_plasma_ramp_required = vs_plasma_ramp_required
    physics.itart = 0
    physics.itartpf = 0

    cs_fatigue = CsFatigue()
    cs_fatigue.data = data
    cs_coil = CSCoil(cs_fatigue=cs_fatigue)
    cs_coil.data = data
    model = PFCoil(cs_fatigue=cs_fatigue, cs_coil=cs_coil)
    model.data = data
    model.pfcoil()

    return (
        p.m_pf_coil_conductor_total,
        p.m_pf_coil_structure_total,
        p.r_pf_coil_outer[: PLASMA_INDEX + 1],
        p.m_pf_coil_conductor[:N_CS_PF_COILS],
        p.m_pf_coil_structure[:N_CS_PF_COILS],
        p.pfcaseth[:N_CS_PF_COILS],
        p.m_pf_coil_max,
        p.ricpf,
        p.a_cs_steel_poloidal,
        p.a_cs_cable_space,
        p.r_pf_coil_outer_max,
        p.n_pf_coil_turns[: PLASMA_INDEX + 1],
        p.r_pf_coil_inner[: PLASMA_INDEX + 1],
        p.z_pf_coil_upper[: PLASMA_INDEX + 1],
        p.z_pf_coil_lower[: PLASMA_INDEX + 1],
        p.r_pf_coil_middle[:N_CS_PF_COILS],
        p.z_pf_coil_middle[:N_CS_PF_COILS],
        p.ccl0,
        p.ccls,
        p.ssq0,
        physics.b_plasma_vertical_required,
        p.f_j_cs_start_end_flat_top,
        p.c_pf_cs_coil_pulse_start_ma[:N_CS_PF_COILS],
        p.c_pf_cs_coil_flat_top_ma[:N_CS_PF_COILS],
        p.c_pf_cs_coil_pulse_end_ma[:N_CS_PF_COILS],
        p.c_pf_cs_coils_peak_ma[:N_CS_PF_COILS],
        p.b_pf_coil_peak[:N_PF_COILS],
        p.bpf2[:N_PF_COILS],
        p.r_pf_outside_tf_midplane,
        p.a_cs_poloidal,
        p.dz_cs_full,
        p.c_pf_coil_turn[: PLASMA_INDEX + 1],
    )


_LEGACY = {
    "z_tf_inside_half": 8.818217164127494,
    "f_z_cs_tf_internal": 0.9,
    "dr_cs": 0.546816593988753,
    "dr_cs_bore": 2.003843190236783,
    "z_tf_top": 8.784333333333333,
    "dz_tf_upper_lower_midplane": -1.233883830794161,
    "rpf2": -1.825,
    "zref": np.array([3.6, 1.2, 1.0, 2.8]),
    "r_tf_outboard_out": 15.578406000060053,
    "dr_pf_tf_outboard_out_offset": 1.5,
    "rmajor": 8.0,
    "rminor": 2.6666666666666665,
    "kappa": 1.85,
    "triang": 0.5,
    "aspect": 3.0,
    "plasma_current": 16091095.408042267,
    "beta_poloidal_vol_avg": 1.3282238170008043,
    "ind_plasma_internal_norm": 1.2568268843995554,
    "vs_plasma_ramp_required": 279.0949824023401,
    "alfapf": 5e-10,
    "j_cs_flat_top_end": 21443595.371072624,
    "f_j_cs_start_pulse_end_flat_top": 0.93491189654662,
    "ind_pf_cs_plasma_mutual_column": np.array([
        0.0007917526382760364,
        0.000730410150563103,
        0.001496022850886937,
        0.001496022850886937,
        0.000730867760781803,
        0.000730867760781803,
    ]),
    "n_pf_coil_turns_previous": np.array([
        463.88982745360926,
        532.1251000998008,
        191.84403916641483,
        191.84403916641483,
        123.23569673015545,
        123.23569673015545,
    ]),
    "j_pf_coil_wp_peak": np.array([
        11000000.0,
        11000000.0,
        6000000.0,
        6000000.0,
        8000000.0,
        8000000.0,
        8000000.0,
    ]),
    "c_pf_coil_turn_peak_input": np.full(7, 40000.0),
    "pf_current_safety_factor": 1.0,
    "f_a_pf_coil_void": np.full(6, 0.3),
    "sigpfcf": 0.666,
    "sigpfcalw": 500.0,
    "den_steel": 7800.0,
    "den_pf_conductor": 6070.0,
    "den_cs_conductor": 6080.0,
    "f_a_cs_turn_steel": 0.4856940627014451,
    "f_a_cs_void": 0.3,
}
"""Every declared input of the chain, read off a converged in-process PROCESS run of
`tests/regression/input_files/large_tokamak_eval.IN.DAT`. At this point PROCESS reports
`m_pf_coil_conductor_total = 2467413.96 kg`, `m_pf_coil_structure_total = 1858967.50 kg`
and `max(r_pf_coil_outer) = 17.434 m`, and re-running `pfcoil()` on it is idempotent --
which is what makes the loop-carried inputs above self-consistent."""


def _fuzz_bounds(fraction=0.10):
    """`+-fraction` around every legacy input, elementwise and sign-safe.

    `zref` and `f_z_cs_tf_internal` are excluded and held fixed: the first sets the
    `i_pf_location = 3` coil heights, which have to stay inside
    `r_pf_outside_tf_midplane` for the D-shaped radius `sqrt(r^2 - z^2)` to be real, and
    the second is a fraction PROCESS reads as a fraction. Both are exercised by
    `test_geometry.py`'s own contracts, which can bound them without dragging the whole
    machine along.
    """
    bounds = {}
    for name, value in _LEGACY.items():
        if name in {"zref", "f_z_cs_tf_internal"}:
            continue
        base = np.asarray(value, dtype=float)
        low = base * (1.0 - fraction)
        high = base * (1.0 + fraction)
        bounds[name] = (np.minimum(low, high), np.maximum(low, high))
    return bounds


class TestPFCoilChain(Tier1Contract):
    """The package's blocks composed, against `PFCoil.pfcoil()` itself."""

    audit_record = "models/pfcoil/masses.md"
    reference = _reference_pf_coil_chain
    ported = _ported_pf_coil_chain
    reference_domain_errors = (ProcessValueError,)

    value_tolerance = Tolerance(
        rtol=5e-11,
        atol=0.0,
        reason=(
            "the chain runs the SVD current solve twice and then propagates its answer "
            "through the coil sizing, the Green's-function peak field and the mass "
            "sums, so the ~2e-13 relative disagreement between scipy's and jax's "
            "LAPACK drivers (see test_currents.py) is amplified by the later stages "
            "rather than absorbed; measured worst component 3e-13 at the reference "
            "point. No solver iterates on either side, so this is still round-off, not "
            "convergence noise"
        ),
    )

    samples = [legacy_sample("large-tokamak-converged", **_LEGACY)]

    fuzz_fixed = {
        "zref": _LEGACY["zref"],
        "f_z_cs_tf_internal": _LEGACY["f_z_cs_tf_internal"],
    }
    fuzz_bounds = _fuzz_bounds()
