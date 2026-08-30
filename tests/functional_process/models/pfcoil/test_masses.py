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
    SPHERICAL_TOKAMAK_TOPOLOGY,
)
from functional_process.models.pfcoil.currents import (
    calculate_cs_flux_swing,
    calculate_equilibrium_currents,
    calculate_plasma_initiation_currents,
    calculate_plasma_initiation_currents_no_central_solenoid,
    calculate_time_point_currents,
    calculate_time_point_currents_no_central_solenoid,
)
from functional_process.models.pfcoil.fields import (
    calculate_coil_current_waveform,
    calculate_pf_coil_peak_fields,
    calculate_pf_coil_peak_fields_no_central_solenoid,
)
from functional_process.models.pfcoil.geometry import (
    calculate_cs_geometry,
    calculate_pf_coil_group_positions,
    calculate_pf_coil_positions,
)
from functional_process.models.pfcoil.masses import (
    calculate_pf_coil_masses,
    calculate_pf_coil_masses_no_central_solenoid,
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


_DCOND_POISON = -1.0e9
"""Seeded into every `.tfcoil.dcond` element the occupant under test does not bind.

The two superconductor switches' only effect in this closure is which `dcond` element
each conductor density is read from (`masses.md` § switches touched), and four of the
nine elements share the value 6080 kg/m^3 -- so a reference that left the array at its
defaults could not tell `dcond[0]` (ITER Nb3Sn) from `dcond[4]` (WST Nb3Sn) at the
legacy point. Poisoning every unbound element makes any wrong-element read a conductor
mass that is wrong in sign and magnitude at *every* sample: the occupant's read has to
move with the switch, which is the `CoilsMass` lesson (`_audit/next_steps.md` §14.11)
turned into an assertion. Safe because `pfcoil()` reads `dcond` in exactly two places
(`pfcoil.py:947`, `:3571`), both switch-indexed."""


def _run_reference_pf_coil_chain(
    i_cs_superconductor,
    cs_dcond_index,
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

    The two leading arguments are not inputs of the chain but the identity of the
    occupant under test: which `i_cs_superconductor` value the reference is driven at,
    and which `dcond` element `den_cs_conductor` is therefore planted in (every other
    element is `_DCOND_POISON`). The two `_reference_pf_coil_chain*` wrappers below
    bind them; the harness never sees them.

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
    p.i_cs_superconductor = i_cs_superconductor
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
    data.tfcoil.dcond = np.full(
        np.asarray(data.tfcoil.dcond).shape, _DCOND_POISON, dtype=float
    )
    data.tfcoil.dcond[2] = den_pf_conductor
    data.tfcoil.dcond[cs_dcond_index] = den_cs_conductor

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


def _reference_pf_coil_chain(**inputs):
    """The reference pair: `i_cs_superconductor = 1` (ITER Nb3Sn), CS density from
    `dcond[0]` -- `large_tokamak_eval.IN.DAT`'s configuration, arm 0.
    """
    return _run_reference_pf_coil_chain(1, 0, **inputs)


def _reference_pf_coil_chain_cs_wst_nb3sn(**inputs):
    """`low_aspect_ratio_DEMO.IN.DAT`'s pair (`:806`, `:845`): `i_cs_superconductor
    = 5` (WST Nb3Sn), CS density from `dcond[4]` -- arm 1,
    `masses.PFCoilMassesCsWstNb3Sn`'s binding.
    """
    return _run_reference_pf_coil_chain(5, 4, **inputs)


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


class TestPFCoilChainCsWstNb3Sn(Tier1Contract):
    """The same chain against `pfcoil()` at `i_cs_superconductor = 5` (WST Nb3Sn CS).

    The occupant under test is `masses.PFCoilMassesCsWstNb3Sn`, whose entire difference
    from `PFCoilMasses` is reading `.tfcoil.dcond[4]` instead of `.tfcoil.dcond[0]` as
    the CS conductor density -- so the ported side is unchanged and the discrimination
    is all in the reference: PROCESS runs with the switch at 5, `den_cs_conductor` is
    planted in `dcond[4]` only, and every other element is `_DCOND_POISON`. If PROCESS
    at this switch value read any element but `[4]` -- or if the occupant's binding
    were the baked `dcond[0]` a `FromExactly` default would silently keep
    (`_audit/next_steps.md` §14.11, the `CoilsMass` lesson) -- every sample would
    disagree, in sign as well as magnitude.

    The inputs are the same converged `large_tokamak_eval` point: the chain is a pure
    function and the (3, 5) pair does not change its domain, only which array slot one
    scalar comes from. `den_cs_conductor = 6080` is also `dcond[4]`'s true value.
    """

    audit_record = "models/pfcoil/masses.md"
    reference = _reference_pf_coil_chain_cs_wst_nb3sn
    ported = _ported_pf_coil_chain
    reference_domain_errors = (ProcessValueError,)

    value_tolerance = Tolerance(
        rtol=5e-11,
        atol=0.0,
        reason=(
            "identical chain to TestPFCoilChain -- the switch changes which dcond "
            "element is read, not any arithmetic -- so the same LAPACK-driver "
            "round-off argument and the same measured headroom apply"
        ),
    )

    samples = [legacy_sample("large-tokamak-point-cs-wst-nb3sn", **_LEGACY)]

    fuzz_fixed = {
        "zref": _LEGACY["zref"],
        "f_z_cs_tf_internal": _LEGACY["f_z_cs_tf_internal"],
    }
    fuzz_bounds = _fuzz_bounds()


# ---------------------------------------------------------------------------
# The same contract for a machine with **no central solenoid** -- `iohcl = 0`,
# `i_pf_location = (2, 3, 3, 4)`, `n_pf_coils_in_group = (2, 2, 2, 2)`, the spherical
# tokamaks' PF coil system (`indat._pf_coil_system_arm` arm 2). Added 2026-08-30.
#
# One reference and one ported side again, and for the same reason: none of these blocks
# is a PROCESS callable, so the oracle has to be `pfcoil()` itself, driven at `iohcl = 0`
# with the eight-coil topology. What it covers that `TestPFCoilChain` cannot:
# `geometry.calculate_pf_coil_group_positions`' `i_pf_location = 4` arm and its
# stacked-radius arm, `calculate_pf_coil_positions` with no CS slot,
# `currents.calculate_plasma_initiation_currents_no_central_solenoid`,
# `calculate_equilibrium_currents` with one fixed-current group and three solved-for,
# `calculate_time_point_currents_no_central_solenoid`,
# `fields.calculate_pf_coil_peak_fields_no_central_solenoid`,
# `masses.calculate_pf_coil_sizes` on a topology with no CS entry, and
# `masses.calculate_pf_coil_masses_no_central_solenoid`.
# ---------------------------------------------------------------------------

I_PF_LOCATION_SPHERICAL_TOKAMAK = np.array([2, 3, 3, 4, 0, 0, 0, 0, 0, 0])
"""`.pf_coil.i_pf_location` on both spherical tokamaks -- one pair above the TF coil,
two pairs outside it, one generally placed."""


def _ported_pf_coil_chain_no_central_solenoid(
    z_tf_top,
    dz_tf_upper_lower_midplane,
    rpf2,
    zref,
    rref,
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
    alfapf,
    f_j_cs_start_pulse_end_flat_top,
    j_pf_coil_wp_peak,
    c_pf_coil_turn_peak_input,
    pf_current_safety_factor,
    f_a_pf_coil_void,
    sigpfcf,
    sigpfcalw,
    den_steel,
    den_pf_conductor,
):
    """`_ported_pf_coil_chain`'s sibling, composed in the same order at `iohcl = 0`.

    Ten declared inputs fewer, which is the whole point of the arm: nothing here reads
    the CS's geometry, its current density, its steel fraction or its conductor density,
    because `ohcalc` is never entered on this machine and no CS field has a producer.
    """
    n = SPHERICAL_TOKAMAK_TOPOLOGY.n_pf_coils
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
        rref=rref,
        topology=SPHERICAL_TOKAMAK_TOPOLOGY,
        r_pf_outside_tf_is_constant=True,
    )
    r_pf_coil_middle, z_pf_coil_middle = calculate_pf_coil_positions(
        r_pf_coil_middle_group_array=r_group,
        z_pf_coil_middle_group_array=z_group,
        topology=SPHERICAL_TOKAMAK_TOPOLOGY,
    )

    ssq0, ccl0 = calculate_plasma_initiation_currents_no_central_solenoid(
        rmajor=rmajor,
        rminor=rminor,
        r_pf_coil_middle_group_array=r_group,
        z_pf_coil_middle_group_array=z_group,
        alfapf=alfapf,
        topology=SPHERICAL_TOKAMAK_TOPOLOGY,
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
        topology=SPHERICAL_TOKAMAK_TOPOLOGY,
    )
    (
        c_start,
        c_flat,
        c_end,
        f_j_cs_start_end_flat_top,
    ) = calculate_time_point_currents_no_central_solenoid(
        ccl0=ccl0,
        ccls=ccls,
        f_j_cs_start_pulse_end_flat_top=f_j_cs_start_pulse_end_flat_top,
        topology=SPHERICAL_TOKAMAK_TOPOLOGY,
    )
    c_peak, f_c_peak_time = calculate_coil_current_waveform(c_start, c_flat, c_end)

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
        r_cs_inner=None,
        r_cs_outer=None,
        z_cs_upper=None,
        z_cs_lower=None,
        rmajor=rmajor,
        rminor=rminor,
        kappa=kappa,
        topology=SPHERICAL_TOKAMAK_TOPOLOGY,
    )

    b_pf_coil_peak, bpf2 = calculate_pf_coil_peak_fields_no_central_solenoid(
        c_pf_cs_coil_pulse_start_ma=c_start,
        c_pf_cs_coil_flat_top_ma=c_flat,
        c_pf_cs_coil_pulse_end_ma=c_end,
        r_pf_coil_middle=r_pf_coil_middle,
        z_pf_coil_middle=z_pf_coil_middle,
        r_pf_coil_inner=r_pf_coil_inner[:n],
        r_pf_coil_outer=r_pf_coil_outer[:n],
        z_pf_coil_upper=z_pf_coil_upper[:n],
        z_pf_coil_lower=z_pf_coil_lower[:n],
        rmajor=rmajor,
        plasma_current=plasma_current,
        topology=SPHERICAL_TOKAMAK_TOPOLOGY,
    )

    (
        m_pf_coil_conductor,
        m_pf_coil_structure,
        pfcaseth,
        m_pf_coil_conductor_total,
        m_pf_coil_structure_total,
        m_pf_coil_max,
        ricpf,
    ) = calculate_pf_coil_masses_no_central_solenoid(
        c_pf_cs_coils_peak_ma=c_peak,
        j_pf_coil_wp_peak=j_pf_coil_wp_peak,
        n_pf_coil_turns=n_pf_coil_turns[:n],
        r_pf_coil_middle=r_pf_coil_middle,
        r_pf_coil_inner=r_pf_coil_inner[:n],
        r_pf_coil_outer=r_pf_coil_outer[:n],
        z_pf_coil_upper=z_pf_coil_upper[:n],
        z_pf_coil_lower=z_pf_coil_lower[:n],
        b_pf_coil_peak=b_pf_coil_peak,
        bpf2=bpf2,
        f_a_pf_coil_void=f_a_pf_coil_void,
        pf_current_safety_factor=pf_current_safety_factor,
        sigpfcf=sigpfcf,
        sigpfcalw=sigpfcalw,
        den_steel=den_steel,
        den_pf_conductor=den_pf_conductor,
        topology=SPHERICAL_TOKAMAK_TOPOLOGY,
    )

    return (
        m_pf_coil_conductor_total,
        m_pf_coil_structure_total,
        m_pf_coil_conductor,
        m_pf_coil_structure,
        pfcaseth,
        m_pf_coil_max,
        ricpf,
        r_pf_coil_outer_max,
        n_pf_coil_turns,
        r_pf_coil_inner,
        r_pf_coil_outer,
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
        f_c_peak_time,
        b_pf_coil_peak,
        bpf2,
        r_pf_outside_tf_midplane,
    )


def _reference_pf_coil_chain_no_central_solenoid(
    z_tf_top,
    dz_tf_upper_lower_midplane,
    rpf2,
    zref,
    rref,
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
    alfapf,
    f_j_cs_start_pulse_end_flat_top,
    j_pf_coil_wp_peak,
    c_pf_coil_turn_peak_input,
    pf_current_safety_factor,
    f_a_pf_coil_void,
    sigpfcf,
    sigpfcalw,
    den_steel,
    den_pf_conductor,
):
    """`PFCoil.pfcoil()` at `iohcl = 0` on the spherical tokamaks' topology.

    `.pf_coil.j_cs_flat_top_end` keeps its `pfcoil_variables.py` default `1.85e7`,
    because the guard at `pfcoil.py:358` reads its product with
    `f_j_cs_start_pulse_end_flat_top` and the block behind that guard is the
    plasma-initiation solve -- which *does* run on this arm, with `nfxf = 0` and
    therefore no dependence on either value. That the ported side declares neither and
    still agrees is the check.

    `.tfcoil.dcond[8]` carries the REBCO tape density and every other element is
    `_DCOND_POISON`, the same discrimination `TestPFCoilChainCsWstNb3Sn` uses.
    """
    n = SPHERICAL_TOKAMAK_TOPOLOGY.n_pf_coils
    data = DataStructure()
    build, p, physics = data.build, data.pf_coil, data.physics

    build.iohcl = 0
    build.z_tf_top = z_tf_top
    build.dz_tf_upper_lower_midplane = dz_tf_upper_lower_midplane

    p.n_pf_coil_groups = SPHERICAL_TOKAMAK_TOPOLOGY.n_pf_coil_groups
    p.n_pf_coils_in_group = np.array(
        [*SPHERICAL_TOKAMAK_TOPOLOGY.n_pf_coils_in_group, 0, 0, 0, 0, 0, 0], dtype=int
    )
    p.i_pf_location = I_PF_LOCATION_SPHERICAL_TOKAMAK.copy()
    p.i_pf_conductor = 0
    p.i_pf_current = 1
    p.i_r_pf_outside_tf_placement = 1
    p.i_pf_superconductor = 9
    p.first_call = False

    p.rpf2 = rpf2
    p.zref = np.concatenate([np.asarray(zref, dtype=float), np.ones(6)])
    p.rref = np.concatenate([np.asarray(rref, dtype=float), np.full(6, 7.0)])
    p.dr_pf_tf_outboard_out_offset = dr_pf_tf_outboard_out_offset
    p.alfapf = alfapf
    p.f_j_cs_start_pulse_end_flat_top = f_j_cs_start_pulse_end_flat_top
    p.pf_current_safety_factor = pf_current_safety_factor
    p.sigpfcf = sigpfcf
    p.sigpfcalw = sigpfcalw

    p.j_pf_coil_wp_peak = np.zeros(NGC2)
    p.j_pf_coil_wp_peak[:n] = j_pf_coil_wp_peak
    p.c_pf_coil_turn_peak_input = np.zeros(NGC2)
    p.c_pf_coil_turn_peak_input[:n] = c_pf_coil_turn_peak_input
    p.f_a_pf_coil_void = np.full(NGC2, 0.3)
    p.f_a_pf_coil_void[:n] = f_a_pf_coil_void
    p.ind_pf_cs_plasma_mutual = np.zeros((NGC2, NGC2))
    p.n_pf_coil_turns = np.full(NGC2, 100.0)

    data.superconducting_tfcoil.r_tf_outboard_out = r_tf_outboard_out
    data.fwbs.den_steel = den_steel
    data.tfcoil.dcond = np.full(
        np.asarray(data.tfcoil.dcond).shape, _DCOND_POISON, dtype=float
    )
    data.tfcoil.dcond[8] = den_pf_conductor

    physics.rmajor = rmajor
    physics.rminor = rminor
    physics.kappa = kappa
    physics.triang = triang
    physics.aspect = aspect
    physics.plasma_current = plasma_current
    physics.beta_poloidal_vol_avg = beta_poloidal_vol_avg
    physics.ind_plasma_internal_norm = ind_plasma_internal_norm
    physics.itart = 1
    physics.itartpf = 1

    cs_fatigue = CsFatigue()
    cs_fatigue.data = data
    cs_coil = CSCoil(cs_fatigue=cs_fatigue)
    cs_coil.data = data
    model = PFCoil(cs_fatigue=cs_fatigue, cs_coil=cs_coil)
    model.data = data
    model.pfcoil()

    plasma = SPHERICAL_TOKAMAK_TOPOLOGY.plasma_index
    return (
        p.m_pf_coil_conductor_total,
        p.m_pf_coil_structure_total,
        p.m_pf_coil_conductor[:n],
        p.m_pf_coil_structure[:n],
        p.pfcaseth[:n],
        p.m_pf_coil_max,
        p.ricpf,
        p.r_pf_coil_outer_max,
        p.n_pf_coil_turns[: plasma + 1],
        p.r_pf_coil_inner[: plasma + 1],
        p.r_pf_coil_outer[: plasma + 1],
        p.z_pf_coil_upper[: plasma + 1],
        p.z_pf_coil_lower[: plasma + 1],
        p.r_pf_coil_middle[:n],
        p.z_pf_coil_middle[:n],
        p.ccl0,
        p.ccls,
        p.ssq0,
        physics.b_plasma_vertical_required,
        p.f_j_cs_start_end_flat_top,
        p.c_pf_cs_coil_pulse_start_ma[:n],
        p.c_pf_cs_coil_flat_top_ma[:n],
        p.c_pf_cs_coil_pulse_end_ma[:n],
        p.c_pf_cs_coils_peak_ma[:n],
        p.f_c_pf_cs_peak_time_array[:n],
        p.b_pf_coil_peak[:n],
        p.bpf2[:n],
        p.r_pf_outside_tf_midplane,
    )


_LEGACY_SPHERICAL_TOKAMAK = {
    "z_tf_top": 4.0,
    "dz_tf_upper_lower_midplane": -0.5,
    "rpf2": -1.63,
    "zref": np.array([3.6, 1.2, 2.5, 5.2]),
    "rref": np.array([7.0, 7.0, 7.0, 2.0]),
    "r_tf_outboard_out": 8.3,
    "dr_pf_tf_outboard_out_offset": 1.0,
    "rmajor": 4.5,
    "rminor": 2.5,
    "kappa": 2.8,
    "triang": 0.5,
    "aspect": 1.8,
    "plasma_current": 20.0e6,
    "beta_poloidal_vol_avg": 1.1,
    "ind_plasma_internal_norm": 1.2,
    "alfapf": 5e-10,
    "f_j_cs_start_pulse_end_flat_top": 0.9,
    "j_pf_coil_wp_peak": np.full(8, 1.1e7),
    "c_pf_coil_turn_peak_input": np.full(8, 40000.0),
    "pf_current_safety_factor": 1.0,
    "f_a_pf_coil_void": np.full(8, 0.3),
    "sigpfcf": 0.666,
    "sigpfcalw": 500.0,
    "den_steel": 7800.0,
    "den_pf_conductor": 6200.0,
}
"""A plausible spherical tokamak, **not a converged point**, and the difference is worth
naming.

`rmajor`, `aspect`, `kappa`, `triang`, `rpf2`, `zref` and `rref` are
`spherical_tokamak_eval.IN.DAT`'s own (`:239-244`, `:260`, `:285`, `:291`, `:296`); the
build quantities the PF system reads (`z_tf_top`, `dz_tf_upper_lower_midplane`,
`r_tf_outboard_out`) are chosen consistent with a 4.5 m machine, because neither ST file
converges through this port yet -- both are still refused on the CroCo TF turn -- and
there is therefore no converged run to read them off.

That costs nothing this contract measures. The oracle is PROCESS's own `pfcoil()`
evaluated at exactly these numbers, so what is compared is two functions at one point of
their common domain, and the fuzz sweep moves every one of them by +-10%."""


def _spherical_tokamak_fuzz_bounds(fraction=0.10):
    """`+-fraction` around every legacy input but `zref`, elementwise and sign-safe.

    `zref` is held fixed for the reason `_fuzz_bounds` holds it fixed on the
    conventional arm: it sets the outside-TF and generally-placed coil heights, and this
    contract is measuring the chain rather than the placement's domain.
    """
    bounds = {}
    for name, value in _LEGACY_SPHERICAL_TOKAMAK.items():
        if name == "zref":
            continue
        base = np.asarray(value, dtype=float)
        low = base * (1.0 - fraction)
        high = base * (1.0 + fraction)
        bounds[name] = (np.minimum(low, high), np.maximum(low, high))
    return bounds


class TestPFCoilChainSphericalTokamak(Tier1Contract):
    """The no-central-solenoid chain, against `PFCoil.pfcoil()` at `iohcl = 0`.

    Bit-for-bit at the legacy point, on every one of the twenty-eight returned
    quantities -- including both SVD solves. The tolerance below is still the chain's
    rather than zero, for the reason `TestPFCoilChain`'s gives: the fuzz sweep moves the
    least-squares matrix and scipy's and jax's LAPACK drivers need not agree in the last
    bits everywhere.
    """

    audit_record = "models/pfcoil/masses.md"
    reference = _reference_pf_coil_chain_no_central_solenoid
    ported = _ported_pf_coil_chain_no_central_solenoid
    reference_domain_errors = (ProcessValueError,)

    value_tolerance = Tolerance(
        rtol=5e-11,
        atol=0.0,
        reason=(
            "the same amplification argument as TestPFCoilChain -- two SVD solves "
            "feeding the coil sizing, the Green's-function peak field and the mass "
            "sums. Measured worst component 0.0 at the reference point (all "
            "twenty-eight outputs agree bit for bit), so the headroom is entirely for "
            "the fuzz sweep"
        ),
    )

    samples = [legacy_sample("spherical-tokamak-plausible", **_LEGACY_SPHERICAL_TOKAMAK)]

    fuzz_fixed = {"zref": _LEGACY_SPHERICAL_TOKAMAK["zref"]}
    fuzz_bounds = _spherical_tokamak_fuzz_bounds()
