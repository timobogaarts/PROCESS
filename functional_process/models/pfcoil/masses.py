"""How big each coil has to be, and what it weighs.

Audit record: `functional_process/_audit/units/models/pfcoil/masses.md`. This module
produces the three variables this wave's new consumers ask `.tokamak.pf_coil` for:
`.pf_coil.m_pf_coil_conductor_total` and `.pf_coil.m_pf_coil_structure_total` (read by
`models/structure.py::Structure`) and `.pf_coil.r_pf_coil_outer` (read by
`models/cryostat.py::Cryostat`).

Two units:

- `calculate_pf_coil_sizes` -- `pfcoil()`'s winding-pack geometry loop
  (`process/models/pfcoil.py:737-845`) for `i_pf_location != 1` coils, together with the
  CS's own slot (`:3237-3294`) and the plasma's (`:1067-1079`), so that each of the five
  per-coil arrays it produces has exactly one owner.
- `calculate_pf_coil_masses` -- the mass loop (`:849-1026`), the CS's steel and
  conductor (`:3504-3583`), and the summations (`:1028-1064`).

**What is deliberately not here.** The mass closure needs a coil's steel *area*, which
comes from the JxB hoop force and (for the CS) from `f_a_cs_turn_steel`. It does not
need any critical current, so `superconpf` (`:4641-4926`) and everything it reaches is
outside this closure and stays UNPORTED -- the ported superconductor fits in
`functional_process/models/physics/superconductors.py` are correspondingly *not* used
here, rather than being re-ported. Nor does it need the CS stress chain
(`calculate_cs_hoop_stress`, `calculate_cs_radial_stress`, the axial-stress profile and
`cs_fatigue.ncycle`, `:3403-3499`), which is the only part of `ohcalc` that touches
`scipy.special.ellipk`/`ellipe` and is therefore also the only part that would need a
new JAX primitive. Both exclusions are itemised in `masses.md`.

**The resistive arms are UNPORTED.** `i_pf_conductor = PFConductorModel.RESISTIVE`
changes `m_pf_coil_conductor`'s density, zeroes `areaspf` and `pfcaseth`, and adds
`p_pf_coil_resistive_total_flat_top`/`p_cs_resistive_flat_top`; that is a different
occupant, not a parameter of this one. On `large_tokamak_eval.IN.DAT`
`i_pf_conductor` takes its default `0` = `SUPERCONDUCTING`
(`pfcoil_variables.py:230`), which is the arm baked in below.
"""

import jax.numpy as jnp
from cottax.interfaces.pytree_namespace_module import (
    ExplicitFunction,
    From,
    FromExactly,
    OutputInto,
)

from functional_process.models.pfcoil import (
    CS_INDEX,
    N_PF_COILS,
    NGC2,
    PLASMA_INDEX,
)
from functional_process.paths import fwbs, pf_coil, physics, tfcoil

I_PF_SUPERCONDUCTOR = 3
"""`.pf_coil.i_pf_superconductor` on the reference run
(`large_tokamak_eval.IN.DAT:246`) -- NbTi, `pfcoil_variables.py:260`. It selects
`.tfcoil.dcond[2] = 6070` kg/m^3 and nothing else in this closure."""

I_CS_SUPERCONDUCTOR = 1
"""`.pf_coil.i_cs_superconductor` on the reference run (`:245`) -- ITER Nb3Sn,
`pfcoil_variables.py:256`. Selects `.tfcoil.dcond[0] = 6080` kg/m^3."""

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
    peak = c_pf_cs_coils_peak_ma[:N_PF_COILS]
    area = (
        jnp.abs(peak * 1.0e6 / j_pf_coil_wp_peak[:N_PF_COILS]) * pf_current_safety_factor
    )
    turns_pf = jnp.abs(peak * 1.0e6 / c_pf_coil_turn_peak_input[:N_PF_COILS])

    dx = 0.5 * jnp.sqrt(area)  # square cross-section
    r_mid = r_pf_coil_middle[:N_PF_COILS]
    z_mid = z_pf_coil_middle[:N_PF_COILS]

    r_inner_pf = r_mid - dx
    r_outer_pf = r_mid + dx
    below = z_mid < 0.0
    z_lower_pf = jnp.where(below, z_mid + dx, z_mid - dx)
    z_upper_pf = jnp.where(below, z_mid - dx, z_mid + dx)

    r_pf_coil_outer_max = jnp.max(r_outer_pf)

    turns_cs = (
        1.0e6
        * jnp.abs(c_pf_cs_coils_peak_ma[CS_INDEX])
        / c_pf_coil_turn_peak_input[CS_INDEX]
    )

    n_pf_coil_turns = jnp.concatenate([turns_pf, jnp.stack([turns_cs, jnp.ones(())])])
    r_pf_coil_inner = jnp.concatenate([
        r_inner_pf,
        jnp.stack([jnp.asarray(r_cs_inner), rmajor - rminor]),
    ])
    r_pf_coil_outer = jnp.concatenate([
        r_outer_pf,
        jnp.stack([jnp.asarray(r_cs_outer), rmajor + rminor]),
    ])
    z_pf_coil_upper = jnp.concatenate([
        z_upper_pf,
        jnp.stack([jnp.asarray(z_cs_upper), rminor * kappa]),
    ])
    z_pf_coil_lower = jnp.concatenate([
        z_lower_pf,
        jnp.stack([jnp.asarray(z_cs_lower), -rminor * kappa]),
    ])

    return (
        n_pf_coil_turns,
        r_pf_coil_inner,
        r_pf_coil_outer,
        z_pf_coil_upper,
        z_pf_coil_lower,
        r_pf_coil_outer_max,
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
    """Conductor and steel mass of every coil, and the totals the rest of the plant reads.

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
    peak = c_pf_cs_coils_peak_ma[:N_PF_COILS]
    turns = n_pf_coil_turns[:N_PF_COILS]
    r_mid = r_pf_coil_middle[:N_PF_COILS]

    area = (
        jnp.abs(peak * 1.0e6 / j_pf_coil_wp_peak[:N_PF_COILS]) * pf_current_safety_factor
    )
    aturn = area / turns
    rll = 2.0 * jnp.pi * r_mid * turns
    volpf = aturn * rll

    m_conductor_pf = volpf * den_pf_conductor * (1.0 - f_a_pf_coil_void)

    forcepf = 0.5e6 * (b_pf_coil_peak + bpf2) * jnp.abs(peak) * r_mid
    areaspf = sigpfcf * forcepf / (sigpfcalw * 1.0e6)

    drpdz = (
        r_pf_coil_outer[:N_PF_COILS]
        - r_pf_coil_inner[:N_PF_COILS]
        + jnp.abs(z_pf_coil_upper[:N_PF_COILS] - z_pf_coil_lower[:N_PF_COILS])
    )
    pfcaseth_pf = 0.25 * (-drpdz + jnp.sqrt(drpdz * drpdz + 4.0 * areaspf))
    m_structure_pf = areaspf * 2.0 * jnp.pi * r_mid * den_steel

    m_pf_coil_max = jnp.max(1.0e-3 * (m_conductor_pf + m_structure_pf))

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


class PFCoilSizes(ExplicitFunction):
    """cottax node: `.tokamak.pf_coil.sizes`.

    Occupant for `i_pf_location != 1` on every group (`(2, 2, 3, 3)` here) with
    `iohcl = 1`. The `i_pf_location = 1` arm sizes a coil from the CS's radial thickness
    and the TF bore instead (`pfcoil.py:744-794`), and also *writes*
    `.pf_coil.j_pf_coil_wp_peak`, which this arm only reads -- a genuinely different
    read/write set, hence a different occupant. UNPORTED.

    Owns five per-coil arrays at their full `NGC2` width plus
    `.pf_coil.r_pf_coil_outer_max`. **One edge of this package's SCC**: it reads
    `.pf_coil.c_pf_cs_coils_peak_ma`, whose producer chain runs back through
    `currents.CSFluxSwing`, which reads `.pf_coil.n_pf_coil_turns` from here. See
    `currents.py`'s module docstring.
    """

    n_pf_coil_turns = OutputInto(pf_coil)
    r_pf_coil_inner = OutputInto(pf_coil)
    r_pf_coil_outer = OutputInto(pf_coil)
    z_pf_coil_upper = OutputInto(pf_coil)
    z_pf_coil_lower = OutputInto(pf_coil)
    r_pf_coil_outer_max = OutputInto(pf_coil)

    def __call__(
        self,
        c_pf_cs_coils_peak_ma=From(pf_coil),
        j_pf_coil_wp_peak=From(pf_coil),
        c_pf_coil_turn_peak_input=From(pf_coil),
        r_pf_coil_middle=From(pf_coil),
        z_pf_coil_middle=From(pf_coil),
        pf_current_safety_factor=From(pf_coil),
        r_cs_inner=From(pf_coil),
        r_cs_outer=From(pf_coil),
        z_cs_upper=From(pf_coil),
        z_cs_lower=From(pf_coil),
        rmajor=From(physics),
        rminor=From(physics),
        kappa=From(physics),
    ):
        (
            turns,
            r_inner,
            r_outer,
            z_upper,
            z_lower,
            r_pf_coil_outer_max,
        ) = calculate_pf_coil_sizes(
            c_pf_cs_coils_peak_ma=c_pf_cs_coils_peak_ma[: CS_INDEX + 1],
            j_pf_coil_wp_peak=j_pf_coil_wp_peak[: CS_INDEX + 1],
            c_pf_coil_turn_peak_input=c_pf_coil_turn_peak_input[: CS_INDEX + 1],
            r_pf_coil_middle=r_pf_coil_middle[: CS_INDEX + 1],
            z_pf_coil_middle=z_pf_coil_middle[: CS_INDEX + 1],
            pf_current_safety_factor=pf_current_safety_factor,
            r_cs_inner=r_cs_inner,
            r_cs_outer=r_cs_outer,
            z_cs_upper=z_cs_upper,
            z_cs_lower=z_cs_lower,
            rmajor=rmajor,
            rminor=rminor,
            kappa=kappa,
        )
        pad = jnp.zeros(NGC2)
        return (
            pad.at[: PLASMA_INDEX + 1].set(turns),
            pad.at[: PLASMA_INDEX + 1].set(r_inner),
            pad.at[: PLASMA_INDEX + 1].set(r_outer),
            pad.at[: PLASMA_INDEX + 1].set(z_upper),
            pad.at[: PLASMA_INDEX + 1].set(z_lower),
            r_pf_coil_outer_max,
        )


class PFCoilMasses(ExplicitFunction):
    """cottax node: `.tokamak.pf_coil.masses`.

    Occupant for `i_pf_conductor = SUPERCONDUCTING`, `i_pf_superconductor = 3` (NbTi)
    and `i_cs_superconductor = 1` (ITER Nb3Sn) with `iohcl = 1`. The two superconductor
    switches enter only as the index of a density in `.tfcoil.dcond`, so they are
    `FromExactly`s at a fixed index -- the same shape
    `models/tfcoil/superconducting.py:1499` and
    `models/stellarator/coils/mass.py:224` already use -- and a different material is a
    different occupant, not a different argument.

    Owns `.pf_coil.m_pf_coil_conductor`, `.pf_coil.m_pf_coil_structure` and
    `.pf_coil.pfcaseth` whole, plus the six scalars derived from them. The six per-index
    reads of `.pf_coil.b_pf_coil_peak`/`.pf_coil.bpf2` match `fields.PFCoilPeakField`'s
    per-index `Output`s: index 6 of both arrays belongs to the CS's own self-field, which
    is UNPORTED and which no mass here depends on.
    """

    m_pf_coil_conductor = OutputInto(pf_coil)
    m_pf_coil_structure = OutputInto(pf_coil)
    pfcaseth = OutputInto(pf_coil)
    m_pf_coil_conductor_total = OutputInto(pf_coil)
    m_pf_coil_structure_total = OutputInto(pf_coil)
    m_pf_coil_max = OutputInto(pf_coil)
    ricpf = OutputInto(pf_coil)
    a_cs_steel_poloidal = OutputInto(pf_coil)
    a_cs_cable_space = OutputInto(pf_coil)

    def __call__(
        self,
        c_pf_cs_coils_peak_ma=From(pf_coil),
        j_pf_coil_wp_peak=From(pf_coil),
        n_pf_coil_turns=From(pf_coil),
        r_pf_coil_middle=From(pf_coil),
        r_pf_coil_inner=From(pf_coil),
        r_pf_coil_outer=From(pf_coil),
        z_pf_coil_upper=From(pf_coil),
        z_pf_coil_lower=From(pf_coil),
        b_pf_coil_peak_0=FromExactly(pf_coil.b_pf_coil_peak[0]),
        b_pf_coil_peak_1=FromExactly(pf_coil.b_pf_coil_peak[1]),
        b_pf_coil_peak_2=FromExactly(pf_coil.b_pf_coil_peak[2]),
        b_pf_coil_peak_3=FromExactly(pf_coil.b_pf_coil_peak[3]),
        b_pf_coil_peak_4=FromExactly(pf_coil.b_pf_coil_peak[4]),
        b_pf_coil_peak_5=FromExactly(pf_coil.b_pf_coil_peak[5]),
        bpf2_0=FromExactly(pf_coil.bpf2[0]),
        bpf2_1=FromExactly(pf_coil.bpf2[1]),
        bpf2_2=FromExactly(pf_coil.bpf2[2]),
        bpf2_3=FromExactly(pf_coil.bpf2[3]),
        bpf2_4=FromExactly(pf_coil.bpf2[4]),
        bpf2_5=FromExactly(pf_coil.bpf2[5]),
        f_a_pf_coil_void=From(pf_coil),
        pf_current_safety_factor=From(pf_coil),
        sigpfcf=From(pf_coil),
        sigpfcalw=From(pf_coil),
        den_steel=From(fwbs),
        den_pf_conductor=FromExactly(tfcoil.dcond[I_PF_SUPERCONDUCTOR - 1]),
        den_cs_conductor=FromExactly(tfcoil.dcond[I_CS_SUPERCONDUCTOR - 1]),
        a_cs_poloidal=From(pf_coil),
        f_a_cs_turn_steel=From(pf_coil),
        f_a_cs_void=From(pf_coil),
    ):
        b_peak = jnp.stack([
            b_pf_coil_peak_0,
            b_pf_coil_peak_1,
            b_pf_coil_peak_2,
            b_pf_coil_peak_3,
            b_pf_coil_peak_4,
            b_pf_coil_peak_5,
        ])
        b_outer = jnp.stack([bpf2_0, bpf2_1, bpf2_2, bpf2_3, bpf2_4, bpf2_5])

        (
            m_conductor,
            m_structure,
            pfcaseth,
            m_conductor_total,
            m_structure_total,
            m_pf_coil_max,
            ricpf,
            a_cs_steel_poloidal,
            a_cs_cable_space,
        ) = calculate_pf_coil_masses(
            c_pf_cs_coils_peak_ma=c_pf_cs_coils_peak_ma[: CS_INDEX + 1],
            j_pf_coil_wp_peak=j_pf_coil_wp_peak[: CS_INDEX + 1],
            n_pf_coil_turns=n_pf_coil_turns[: CS_INDEX + 1],
            r_pf_coil_middle=r_pf_coil_middle[: CS_INDEX + 1],
            r_pf_coil_inner=r_pf_coil_inner[: CS_INDEX + 1],
            r_pf_coil_outer=r_pf_coil_outer[: CS_INDEX + 1],
            z_pf_coil_upper=z_pf_coil_upper[: CS_INDEX + 1],
            z_pf_coil_lower=z_pf_coil_lower[: CS_INDEX + 1],
            b_pf_coil_peak=b_peak,
            bpf2=b_outer,
            f_a_pf_coil_void=f_a_pf_coil_void[:N_PF_COILS],
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
        pad = jnp.zeros(NGC2)
        return (
            pad.at[: CS_INDEX + 1].set(m_conductor),
            pad.at[: CS_INDEX + 1].set(m_structure),
            pad.at[: CS_INDEX + 1].set(pfcaseth),
            m_conductor_total,
            m_structure_total,
            m_pf_coil_max,
            ricpf,
            a_cs_steel_poloidal,
            a_cs_cable_space,
        )
