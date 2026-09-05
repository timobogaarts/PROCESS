"""Pure functions for the PF coil sizing (turns, cross-section, edges) and the
conductor/steel masses this pass's boundary asks for, extracted from
`functional_process/cottax/pfcoil/masses.py`.

That module still holds the graph declarations (`ExplicitFunction` occupants) that wire
these functions to `VarPath`s; read its module docstring (and
`functional_process/cottax/pfcoil/__init__.py`'s package docstring) for scope. The
audit record is `functional_process/_audit/units/models/pfcoil/masses.md` and mirrors
these functions, not the declarations that call them.
"""

import jax.numpy as jnp

from functional_process.models.pfcoil import (
    CS_INDEX,
    N_PF_COILS,
    NGC2,
    REFERENCE_TOPOLOGY,
    SPHERICAL_TOKAMAK_TOPOLOGY,
)
from functional_process.models.safe_math import safe_sqrt

_A_CS_CABLE_SPACE_FLOOR = 1.0e-4
"""`da`, 1 cm^2 (`pfcoil.py:3555`). Issue #97's fudge keeps the CS cable space positive
with a continuous, smooth, monotonically decreasing replacement below the floor."""


def calculate_pf_coil_sizes(
    c_pf_cs_coils_peak_ma,
    j_pf_coil_wp_peak,
    c_pf_coil_turn_peak_input,
    r_pf_coil_middle,
    z_pf_coil_middle,
    pf_current_safety_factor,
    r_cs_inner,
    r_cs_outer,
    z_cs_upper,
    z_cs_lower,
    rmajor,
    rminor,
    kappa,
    *,
    topology=REFERENCE_TOPOLOGY,
):
    """Winding-pack cross-section, turn count and edge coordinates for every coil.

    Ports the `else` arm of `pfcoil()`'s geometry loop
    (`process/models/pfcoil.py:796-837`) -- "other coils", i.e. everything that is not
    `i_pf_location = 1`, which on this run is all six PF coils. Their winding pack is
    taken square, sized from the peak current and the input current density, and the
    coil's edges follow from its centre.

    The CS's own five entries come from `ohcalc` (`:3237-3294`) and the plasma's from
    `pfcoil()`'s closing block (`:1067-1079`); both are folded in so that each returned
    array has a single owner rather than three nodes each claiming a slice.

    PROCESS's `if z_pf_coil_middle[i] < 0` sign flips (`:826-837`) swap which edge is
    called "upper" and which "lower" for a coil below the midplane. Kept exactly:
    `z_pf_coil_upper` is the edge *further* from the midplane, whatever its sign.

    Parameters
    ----------
    c_pf_cs_coils_peak_ma :
        Peak current in each coil (MA), seven entries -- six PF coils then the CS.
    j_pf_coil_wp_peak :
        Winding-pack current density at peak current (A/m^2), an input for these coils
        ("defined in routine INITIAL", `pfcoil.py:797-798`).
    c_pf_coil_turn_peak_input :
        Peak current per turn (A), an input.
    r_pf_coil_middle, z_pf_coil_middle :
        Coil centres (m), seven entries.
    pf_current_safety_factor :
        Multiplier on the winding-pack area. `.pf_coil.pf_current_safety_factor`.
    r_cs_inner, r_cs_outer, z_cs_upper, z_cs_lower :
        The CS's own edges (m), from `geometry.calculate_cs_geometry`.
    rmajor, rminor, kappa :
        Plasma major/minor radius (m) and elongation, for the plasma's slot.

    Returns
    -------
    tuple
        `(n_pf_coil_turns, r_pf_coil_inner, r_pf_coil_outer, z_pf_coil_upper,
        z_pf_coil_lower, r_pf_coil_outer_max)` -- the five arrays eight entries long
        (six PF coils, the CS, the plasma) and the largest PF coil outer radius (m).
    """
    n = topology.n_pf_coils
    peak = c_pf_cs_coils_peak_ma[:n]
    area = jnp.abs(peak * 1.0e6 / j_pf_coil_wp_peak[:n]) * pf_current_safety_factor
    turns_pf = jnp.abs(peak * 1.0e6 / c_pf_coil_turn_peak_input[:n])

    # Square cross-section. `safe_sqrt`, not `jnp.sqrt`: `area` is zero whenever
    # `pf_current_safety_factor` or a coil's peak current is, and `sqrt`'s derivative
    # there is `inf` while its value is correct -- `_audit/next_steps.md` §9's trap, and
    # `test_gradient_finite_at_zero` catches it at exactly that point.
    dx = 0.5 * safe_sqrt(area)
    r_mid = r_pf_coil_middle[:n]
    z_mid = z_pf_coil_middle[:n]

    r_inner_pf = r_mid - dx
    r_outer_pf = r_mid + dx
    below = z_mid < 0.0
    z_lower_pf = jnp.where(below, z_mid + dx, z_mid - dx)
    z_upper_pf = jnp.where(below, z_mid - dx, z_mid + dx)

    r_pf_coil_outer_max = jnp.max(r_outer_pf)

    turns_tail = [jnp.ones(())]
    r_inner_tail = [rmajor - rminor]
    r_outer_tail = [rmajor + rminor]
    z_upper_tail = [rminor * kappa]
    z_lower_tail = [-rminor * kappa]
    if topology.has_central_solenoid:
        cs = topology.cs_index
        turns_tail.insert(
            0,
            1.0e6 * jnp.abs(c_pf_cs_coils_peak_ma[cs]) / c_pf_coil_turn_peak_input[cs],
        )
        r_inner_tail.insert(0, jnp.asarray(r_cs_inner))
        r_outer_tail.insert(0, jnp.asarray(r_cs_outer))
        z_upper_tail.insert(0, jnp.asarray(z_cs_upper))
        z_lower_tail.insert(0, jnp.asarray(z_cs_lower))

    n_pf_coil_turns = jnp.concatenate([turns_pf, jnp.stack(turns_tail)])
    r_pf_coil_inner = jnp.concatenate([r_inner_pf, jnp.stack(r_inner_tail)])
    r_pf_coil_outer = jnp.concatenate([r_outer_pf, jnp.stack(r_outer_tail)])
    z_pf_coil_upper = jnp.concatenate([z_upper_pf, jnp.stack(z_upper_tail)])
    z_pf_coil_lower = jnp.concatenate([z_lower_pf, jnp.stack(z_lower_tail)])

    return (
        n_pf_coil_turns,
        r_pf_coil_inner,
        r_pf_coil_outer,
        z_pf_coil_upper,
        z_pf_coil_lower,
        r_pf_coil_outer_max,
    )


def _pf_coil_masses_per_coil(
    c_pf_cs_coils_peak_ma,
    j_pf_coil_wp_peak,
    n_pf_coil_turns,
    r_pf_coil_middle,
    r_pf_coil_inner,
    r_pf_coil_outer,
    z_pf_coil_upper,
    z_pf_coil_lower,
    b_pf_coil_peak,
    bpf2,
    f_a_pf_coil_void,
    pf_current_safety_factor,
    sigpfcf,
    sigpfcalw,
    den_steel,
    den_pf_conductor,
    n_pf_coils,
):
    """The per-PF-coil half of `pfcoil()`'s mass loop (`pfcoil.py:849-1026`).

    Everything the loop does for an `i_pf_location != 1` superconducting coil, and
    nothing the central solenoid needs -- so it is the same body on both arms, and the
    two callers differ only in what they append to it.

    Returns
    -------
    tuple
        `(m_pf_coil_conductor, m_pf_coil_structure, pfcaseth, m_pf_coil_max)` -- three
        arrays `n_pf_coils` long (kg, kg, m) and the heaviest coil (tonnes).
    """
    peak = c_pf_cs_coils_peak_ma[:n_pf_coils]
    turns = n_pf_coil_turns[:n_pf_coils]
    r_mid = r_pf_coil_middle[:n_pf_coils]

    area = (
        jnp.abs(peak * 1.0e6 / j_pf_coil_wp_peak[:n_pf_coils]) * pf_current_safety_factor
    )
    aturn = area / turns
    rll = 2.0 * jnp.pi * r_mid * turns
    volpf = aturn * rll

    m_conductor_pf = volpf * den_pf_conductor * (1.0 - f_a_pf_coil_void)

    forcepf = 0.5e6 * (b_pf_coil_peak + bpf2) * jnp.abs(peak) * r_mid
    areaspf = sigpfcf * forcepf / (sigpfcalw * 1.0e6)

    drpdz = (
        r_pf_coil_outer[:n_pf_coils]
        - r_pf_coil_inner[:n_pf_coils]
        + jnp.abs(z_pf_coil_upper[:n_pf_coils] - z_pf_coil_lower[:n_pf_coils])
    )
    pfcaseth_pf = 0.25 * (-drpdz + jnp.sqrt(drpdz * drpdz + 4.0 * areaspf))
    m_structure_pf = areaspf * 2.0 * jnp.pi * r_mid * den_steel

    m_pf_coil_max = jnp.max(1.0e-3 * (m_conductor_pf + m_structure_pf))
    return m_conductor_pf, m_structure_pf, pfcaseth_pf, m_pf_coil_max


def calculate_pf_coil_masses_no_central_solenoid(
    c_pf_cs_coils_peak_ma,
    j_pf_coil_wp_peak,
    n_pf_coil_turns,
    r_pf_coil_middle,
    r_pf_coil_inner,
    r_pf_coil_outer,
    z_pf_coil_upper,
    z_pf_coil_lower,
    b_pf_coil_peak,
    bpf2,
    f_a_pf_coil_void,
    pf_current_safety_factor,
    sigpfcf,
    sigpfcalw,
    den_steel,
    den_pf_conductor,
    *,
    topology=SPHERICAL_TOKAMAK_TOPOLOGY,
):
    """`calculate_pf_coil_masses` on a machine with no central solenoid.

    Ports `pfcoil()`'s per-coil mass loop (`process/models/pfcoil.py:849-1026`) and its
    summations (`:1052-1064`) at `iohcl = 0`, `i_pf_conductor = SUPERCONDUCTING`.
    `ohcalc`'s CS steel and conductor block (`:3504-3583`) is **not** ported here and
    not replaced by zeros: `ohcalc` is not entered at all on this arm
    (`:1048-1050`), so `.pf_coil.a_cs_steel_poloidal` and `.pf_coil.a_cs_cable_space`
    have no producer -- absence, spelled as absence, exactly as
    `models/tokamak/namespace.py`'s rule asks. The two conductor densities become one:
    `den_cs_conductor` is not read, because there is no CS conductor to weigh.

    Returns
    -------
    tuple
        `(m_pf_coil_conductor, m_pf_coil_structure, pfcaseth,
        m_pf_coil_conductor_total, m_pf_coil_structure_total, m_pf_coil_max, ricpf)` --
        three per-coil arrays `topology.n_pf_coils` long (kg, kg, m), then the totals
        (kg, kg, tonnes, MA).
    """
    (
        m_pf_coil_conductor,
        m_pf_coil_structure,
        pfcaseth,
        m_pf_coil_max,
    ) = _pf_coil_masses_per_coil(
        c_pf_cs_coils_peak_ma=c_pf_cs_coils_peak_ma,
        j_pf_coil_wp_peak=j_pf_coil_wp_peak,
        n_pf_coil_turns=n_pf_coil_turns,
        r_pf_coil_middle=r_pf_coil_middle,
        r_pf_coil_inner=r_pf_coil_inner,
        r_pf_coil_outer=r_pf_coil_outer,
        z_pf_coil_upper=z_pf_coil_upper,
        z_pf_coil_lower=z_pf_coil_lower,
        b_pf_coil_peak=b_pf_coil_peak,
        bpf2=bpf2,
        f_a_pf_coil_void=f_a_pf_coil_void,
        pf_current_safety_factor=pf_current_safety_factor,
        sigpfcf=sigpfcf,
        sigpfcalw=sigpfcalw,
        den_steel=den_steel,
        den_pf_conductor=den_pf_conductor,
        n_pf_coils=topology.n_pf_coils,
    )
    return (
        m_pf_coil_conductor,
        m_pf_coil_structure,
        pfcaseth,
        jnp.sum(m_pf_coil_conductor),
        jnp.sum(m_pf_coil_structure),
        m_pf_coil_max,
        jnp.sum(jnp.abs(c_pf_cs_coils_peak_ma[: topology.n_cs_pf_coils])),
    )


def calculate_pf_coil_masses(
    c_pf_cs_coils_peak_ma,
    j_pf_coil_wp_peak,
    n_pf_coil_turns,
    r_pf_coil_middle,
    r_pf_coil_inner,
    r_pf_coil_outer,
    z_pf_coil_upper,
    z_pf_coil_lower,
    b_pf_coil_peak,
    bpf2,
    f_a_pf_coil_void,
    pf_current_safety_factor,
    sigpfcf,
    sigpfcalw,
    den_steel,
    den_pf_conductor,
    den_cs_conductor,
    a_cs_poloidal,
    f_a_cs_turn_steel,
    f_a_cs_void,
):
    """Conductor and steel mass of every coil, and the totals the plant reads.

    Ports `pfcoil()`'s per-coil mass loop (`process/models/pfcoil.py:849-1026`) at
    `i_pf_conductor = SUPERCONDUCTING`, `ohcalc`'s CS steel and conductor
    (`:3504-3583`), and the summations (`:1028-1064`).

    Everything in the loop that this closure does not need is dropped rather than
    computed and discarded: `superconpf` and `j_pf_wp_critical`/`j_crit_str_pf`
    (`:871-904`), and the resistive-coil power sum (`:917-936`, a different occupant).

    **`.pf_coil.itr_sum` is deliberately not produced here** even though PROCESS computes
    it in the same block (`:1028-1046`). Its CS term reads
    `n_pf_coil_turns[n_cs_pf_coils - 1]`, and that read happens *before* `ohcalc()` runs
    at `:1050` -- so it consumes the **previous** pipeline pass's CS turn count, not the
    one this pass computes. Measured, not inferred: driving PROCESS's own `pfcoil()` from
    a cold `DataStructure` with the reference run's inputs and a seeded
    `n_pf_coil_turns[6] = 100` gives `itr_sum = 6.412e8` where the converged run reports
    `1.056e9`, the difference being exactly
    `(dr_cs_bore + dr_cs/2) * (4653.0 - 100.0) * 4e4`. Producing it faithfully would
    mean declaring a second, distinct loop-carried input; producing it from the current
    turn count would be a silently different number. Neither belongs in a pass whose
    boundary does not ask for it, so it is UNPORTED -- see `masses.md`.

    Parameters
    ----------
    c_pf_cs_coils_peak_ma :
        Peak current in each coil (MA), seven entries.
    j_pf_coil_wp_peak, pf_current_safety_factor :
        As for `calculate_pf_coil_sizes`; the winding-pack area is recomputed from them
        here exactly as PROCESS recomputes it, rather than passed between nodes as a
        local that has no `VarPath`.
    n_pf_coil_turns :
        Turns in each coil, seven entries.
    r_pf_coil_middle, r_pf_coil_inner, r_pf_coil_outer, z_pf_coil_upper,
    z_pf_coil_lower :
        Coil centre and edges (m), seven entries.
    b_pf_coil_peak, bpf2 :
        Field at the inner and outer edge of each PF coil (T), six entries.
    f_a_pf_coil_void :
        Void fraction of each PF coil's winding pack, six entries.
    sigpfcf :
        Fraction of the JxB hoop force carried by the steel case.
    sigpfcalw :
        Allowable stress in that case (MPa).
    den_steel :
        Steel density (kg/m^3). `.fwbs.den_steel`.
    den_pf_conductor, den_cs_conductor :
        Superconductor density (kg/m^3) -- `.tfcoil.dcond[i_pf_superconductor - 1]` and
        `.tfcoil.dcond[i_cs_superconductor - 1]`.
    a_cs_poloidal, f_a_cs_turn_steel, f_a_cs_void :
        CS poloidal cross-section (m^2), its steel fraction and its void fraction.

    Returns
    -------
    tuple
        `(m_pf_coil_conductor, m_pf_coil_structure, pfcaseth,
        m_pf_coil_conductor_total, m_pf_coil_structure_total, m_pf_coil_max, ricpf,
        a_cs_steel_poloidal, a_cs_cable_space)` -- the three per-coil arrays seven
        entries long (kg, kg, m), then the totals (kg, kg, tonnes, MA, m^2, m^2).
    """
    (
        m_conductor_pf,
        m_structure_pf,
        pfcaseth_pf,
        m_pf_coil_max,
    ) = _pf_coil_masses_per_coil(
        c_pf_cs_coils_peak_ma=c_pf_cs_coils_peak_ma,
        j_pf_coil_wp_peak=j_pf_coil_wp_peak,
        n_pf_coil_turns=n_pf_coil_turns,
        r_pf_coil_middle=r_pf_coil_middle,
        r_pf_coil_inner=r_pf_coil_inner,
        r_pf_coil_outer=r_pf_coil_outer,
        z_pf_coil_upper=z_pf_coil_upper,
        z_pf_coil_lower=z_pf_coil_lower,
        b_pf_coil_peak=b_pf_coil_peak,
        bpf2=bpf2,
        f_a_pf_coil_void=f_a_pf_coil_void,
        pf_current_safety_factor=pf_current_safety_factor,
        sigpfcf=sigpfcf,
        sigpfcalw=sigpfcalw,
        den_steel=den_steel,
        den_pf_conductor=den_pf_conductor,
        n_pf_coils=N_PF_COILS,
    )

    # --- Central Solenoid (`ohcalc`, superconducting arm) ---
    r_cs_middle = r_pf_coil_middle[CS_INDEX]
    a_cs_steel_poloidal = f_a_cs_turn_steel * a_cs_poloidal
    pfcaseth_cs = 0.25 * a_cs_steel_poloidal / z_pf_coil_upper[CS_INDEX]
    m_structure_cs = a_cs_steel_poloidal * 2.0 * jnp.pi * r_cs_middle * den_steel

    a_cs_cable_space_raw = a_cs_poloidal - a_cs_steel_poloidal
    da = _A_CS_CABLE_SPACE_FLOOR
    a_cs_cable_space = jnp.where(
        a_cs_cable_space_raw < da,
        da * da / (2.0 * da - a_cs_cable_space_raw),
        a_cs_cable_space_raw,
    )
    m_conductor_cs = (
        a_cs_cable_space
        * (1.0 - f_a_cs_void)
        * 2.0
        * jnp.pi
        * r_cs_middle
        * den_cs_conductor
    )

    m_pf_coil_conductor = jnp.concatenate([
        m_conductor_pf,
        jnp.atleast_1d(m_conductor_cs),
    ])
    m_pf_coil_structure = jnp.concatenate([
        m_structure_pf,
        jnp.atleast_1d(m_structure_cs),
    ])
    pfcaseth = jnp.concatenate([pfcaseth_pf, jnp.atleast_1d(pfcaseth_cs)])

    return (
        m_pf_coil_conductor,
        m_pf_coil_structure,
        pfcaseth,
        jnp.sum(m_pf_coil_conductor),
        jnp.sum(m_pf_coil_structure),
        m_pf_coil_max,
        jnp.sum(jnp.abs(c_pf_cs_coils_peak_ma)),
        a_cs_steel_poloidal,
        a_cs_cable_space,
    )


def calculate_pf_coil_masses_no_central_solenoid_for_topology(
    c_pf_cs_coils_peak_ma,
    j_pf_coil_wp_peak,
    n_pf_coil_turns,
    r_pf_coil_middle,
    r_pf_coil_inner,
    r_pf_coil_outer,
    z_pf_coil_upper,
    z_pf_coil_lower,
    b_pf_coil_peak,
    bpf2,
    f_a_pf_coil_void,
    pf_current_safety_factor,
    sigpfcf,
    sigpfcalw,
    den_steel,
    den_pf_conductor,
    *,
    topology,
):
    """`PFCoilMassesNoCentralSolenoid`: trims every coil array to `topology`'s width
    and pads the three per-coil outputs back out to `NGC2`.
    """
    n = topology.n_pf_coils
    (
        m_conductor,
        m_structure,
        pfcaseth,
        m_conductor_total,
        m_structure_total,
        m_pf_coil_max,
        ricpf,
    ) = calculate_pf_coil_masses_no_central_solenoid(
        c_pf_cs_coils_peak_ma=c_pf_cs_coils_peak_ma[: topology.n_cs_pf_coils],
        j_pf_coil_wp_peak=j_pf_coil_wp_peak[:n],
        n_pf_coil_turns=n_pf_coil_turns[:n],
        r_pf_coil_middle=r_pf_coil_middle[:n],
        r_pf_coil_inner=r_pf_coil_inner[:n],
        r_pf_coil_outer=r_pf_coil_outer[:n],
        z_pf_coil_upper=z_pf_coil_upper[:n],
        z_pf_coil_lower=z_pf_coil_lower[:n],
        b_pf_coil_peak=b_pf_coil_peak[:n],
        bpf2=bpf2[:n],
        f_a_pf_coil_void=f_a_pf_coil_void[:n],
        pf_current_safety_factor=pf_current_safety_factor,
        sigpfcf=sigpfcf,
        sigpfcalw=sigpfcalw,
        den_steel=den_steel,
        den_pf_conductor=den_pf_conductor,
        topology=topology,
    )
    pad = jnp.zeros(NGC2)
    return (
        pad.at[:n].set(m_conductor),
        pad.at[:n].set(m_structure),
        pad.at[:n].set(pfcaseth),
        m_conductor_total,
        m_structure_total,
        m_pf_coil_max,
        ricpf,
    )
