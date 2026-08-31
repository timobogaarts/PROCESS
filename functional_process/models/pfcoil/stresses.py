"""The Central Solenoid's stress state -- `ohcalc`'s superconducting-coil stress block.

Audit record: `functional_process/_audit/units/models/pfcoil/stresses.md`, which the wave
that wrote this module could not create -- `unit_registry.md` was held open by two
sibling agents, so the material went into `pfcoil/fields.md` § "the CS chain" and this
docstring said a row was owed. Both were done on 2026-08-29 and the material moved
unchanged.

Ports `process/models/pfcoil.py:3398-3521` -- the `i_pf_conductor == SUPERCONDUCTING`
arm of `ohcalc`'s stress block: Wilson's hoop and radial stresses, the elliptic-integral
axial self-stress, and the Tresca/von Mises combinations of the three.
`optimise_design.md` §11.5's constraint-72 row: `.pf_coil.stress_shear_cs_peak` was a
boundary zero against PROCESS's converged `1.1647e9` Pa.

**The elliptic integrals are why this block was UNPORTED**, and they are what this
module actually adds. `models/pfcoil/namespace.py::CSCoil` named "`ohcalc`'s
`scipy.special` ellipk/ellipe calls" as the blocker in so many words.
`calculate_cs_self_peak_midplane_axial_stress` calls
`scipy.special.ellipk`/`ellipe`, which are opaque C and have no JAX equivalent --
`jax.scipy.special` does not carry them. `_ellipk`/`_ellipe` below are the
arithmetic-geometric mean, which is traceable, differentiable, and agrees with scipy to
1-2 ulp over the whole unit interval (measured, and pinned by this unit's tier-1 cases).

That is a **different** answer from the one `fields.py`'s Green's-function kernel gives
for the same two functions, and deliberately so: there, PROCESS itself uses Abramowitz &
Stegun's rational fits inline (`pfcoil.py:4969-4986`) and the port transcribes them,
fits and all, because reproducing PROCESS means reproducing its approximation error.
Here PROCESS calls the exact library, so the port must be exact too -- an A&S fit
substituted in this block would be a ~1e-7 divergence dressed as a port. Two ports of
"the elliptic integrals", one per call site, each matching what PROCESS actually
evaluates.

**Not ported from this block**: the 21-point vertical profile of the axial self-stress
(`:3436-3465`, `.pf_coil.stress_z_cs_self_profile`). Nothing in the graph and no active
constraint reads it, and it carries a `np.isnan` sweep that is a data-dependent mask
over a fixed grid -- portable, but not for free, and not for nothing.

**The CS fatigue call (`:3486-3499`) left this list on 2026-08-30.** It is a whole
`Model` of its own, so it is ported as one -- `models/cs_fatigue.py::CsFatigue`, filling
`.tokamak.cs_fatigue`, which was an empty slot when this docstring was written. The
sentence above it used to say "neither is read by any active constraint", and *that* was
the error: constraint 90 reads `.cs_fatigue.n_cycle` and is active on
`low_aspect_ratio_DEMO`, where it was violated by exactly `+1.000000` with a zero
gradient row because nothing owned the field. `stress_hoop_cs_inner`, which this module
owns, is that node's one physics read.
"""

import jax.numpy as jnp
from cottax.interfaces.pytree_namespace_module import (
    ExplicitFunction,
    From,
    OutputInto,
)

from functional_process.models.pfcoil import CS_INDEX
from functional_process.paths import pf_coil, tfcoil
from functional_process.vocabulary import constants

RMU0 = constants.RMU0
"""Vacuum permeability (H/m), imported for the same reason `fields.py` imports it."""

_N_AGM = 12
"""Arithmetic-geometric mean iterations in `_ellipk`/`_ellipe`.

The AGM converges *quadratically* -- the number of correct digits doubles each step --
so twelve is far past double precision for every `m` in `[0, 1)`; measured against
`scipy.special` at `m = 1e-8 ... 0.9999`, the worst relative error is `2.2e-16` for `K`
and `1.1e-15` for `E`. Fixed rather than adaptive because a data-dependent trip count is
not traceable, and there is nothing to gain: the extra iterations cost eight flops each
and the loop is unrolled at trace time.
"""

_RADIAL_TERM_SNAP = 1e-8
"""`np.isclose(x, 0.0)`'s effective threshold in `calculate_cs_radial_stress`
(`pfcoil.py:4402`, `:4408`) -- `np.isclose` against a literal `0.0` reduces to
`|x| <= atol` with numpy's default `atol = 1e-8`, since the `rtol * |0|` term vanishes.

**PROCESS's own snap-to-zero, reproduced rather than removed.** Both guarded terms are
*algebraically* zero when the stress point is the coil's inner radius (`epsilon = 1`
makes `hp_term_2 = alpha^2 + alpha + 1 - alpha^2 - (alpha + 1)` and
`hp_term_4 = alpha^2 + 1 - alpha^2 - 1`), and PROCESS evaluates that expression in
floating point and then snaps the residue. `ohcalc` calls the function at exactly that
point for `.pf_coil.stress_radial_cs_inner` (`:3476-3483`, "In reality this is
practially 0" [sic]), so the guard is not hypothetical -- it is what makes that output
exactly `0.0` instead of ~1e-12."""


def _ellipk(m):
    """Complete elliptic integral of the first kind, `K(m)`, by the AGM.

    `K(m) = pi / (2 * agm(1, sqrt(1 - m)))`. Matches `scipy.special.ellipk`'s
    *parameter* convention (`m = k^2`), which is the one `ohcalc` uses.

    Parameters
    ----------
    m :
        Parameter, `0 <= m < 1`.

    Returns
    -------
    :
        `K(m)`.
    """
    a = jnp.ones_like(jnp.asarray(m, float))
    b = jnp.sqrt(1.0 - m)
    for _ in range(_N_AGM):
        a, b = 0.5 * (a + b), jnp.sqrt(a * b)
    return jnp.pi / (2.0 * a)


def _ellipe(m):
    """Complete elliptic integral of the second kind, `E(m)`, by the AGM.

    `E(m) = K(m) * (1 - sum_n 2^(n-1) c_n^2)` with `c_0 = sqrt(m)` and
    `c_(n+1) = (a_n - b_n) / 2` along the same AGM sequence `K` uses -- so the two share
    one recursion and `E` costs one extra accumulator. Matches
    `scipy.special.ellipe`'s parameter convention.

    Parameters
    ----------
    m :
        Parameter, `0 <= m < 1`.

    Returns
    -------
    :
        `E(m)`.
    """
    m = jnp.asarray(m, float)
    a = jnp.ones_like(m)
    b = jnp.sqrt(1.0 - m)
    total = 0.5 * m
    for n in range(1, _N_AGM + 1):
        a_next = 0.5 * (a + b)
        b_next = jnp.sqrt(a * b)
        total = total + (2.0 ** (n - 1)) * (0.5 * (a - b)) ** 2
        a, b = a_next, b_next
    return (jnp.pi / (2.0 * a)) * (1.0 - total)


def calculate_cs_hoop_stress(
    r_stress_point,
    r_cs_inner,
    r_cs_outer,
    j_cs,
    b_cs_inner,
    f_poisson_cs_structure,
    f_a_cs_turn_steel,
):
    """Hoop stress in the CS at one radius (Pa).

    Ports `CSCoil.calculate_cs_hoop_stress`, `process/models/pfcoil.py:4247-4335`,
    unchanged (M. N. Wilson, *Superconducting Magnets*). The field at the coil's outer
    radius is PROCESS's own hardcoded `0.0` ("Assume to be 0 for now"), folded in rather
    than exposed -- it is a literal in the source, not a variable.

    The final division by `f_a_cs_turn_steel` is the un-smearing step: Wilson's formula
    gives the stress averaged over the whole turn, and the steel carries all of it.

    Parameters
    ----------
    r_stress_point :
        Radius at which the stress is wanted (m).
    r_cs_inner, r_cs_outer :
        CS inner and outer radii (m).
    j_cs :
        CS current density (A/m^2).
    b_cs_inner :
        Field at the CS inner radius (T).
    f_poisson_cs_structure :
        Poisson's ratio of the CS structure.
    f_a_cs_turn_steel :
        Steel area fraction of the CS turn cross-section.

    Returns
    -------
    :
        Hoop stress (Pa).
    """
    alpha = r_cs_outer / r_cs_inner
    epsilon = r_stress_point / r_cs_inner
    b_cs_outer = 0.0

    k = ((alpha * b_cs_inner - b_cs_outer) * j_cs * r_cs_inner) / (alpha - 1.0)
    m = ((b_cs_inner - b_cs_outer) * j_cs * r_cs_inner) / (alpha - 1.0)

    hp_term_1 = k * ((2.0 + f_poisson_cs_structure) / (3.0 * (alpha + 1.0)))
    hp_term_2 = (
        alpha**2
        + alpha
        + 1.0
        + alpha**2 / epsilon**2
        - epsilon
        * (
            ((1.0 + 2.0 * f_poisson_cs_structure) * (alpha + 1.0))
            / (2.0 + f_poisson_cs_structure)
        )
    )
    hp_term_3 = m * ((3.0 + f_poisson_cs_structure) / 8.0)
    hp_term_4 = (
        alpha**2
        + 1.0
        + alpha**2 / epsilon**2
        - epsilon**2
        * ((1.0 + 3.0 * f_poisson_cs_structure) / (3.0 + f_poisson_cs_structure))
    )

    return (hp_term_1 * hp_term_2 - hp_term_3 * hp_term_4) / f_a_cs_turn_steel


def calculate_cs_radial_stress(
    r_stress_point,
    r_cs_inner,
    r_cs_outer,
    j_cs,
    b_cs_inner,
    f_poisson_cs_structure,
):
    """Radial stress in the CS at one radius (Pa).

    Ports `CSCoil.calculate_cs_radial_stress`, `process/models/pfcoil.py:4337-4412`.
    Same `K`/`M` terms as the hoop stress; the two shape terms differ, and both carry
    PROCESS's snap-to-zero guard (see `_RADIAL_TERM_SNAP`). Not un-smeared -- PROCESS
    does not divide this one by the steel fraction, and that asymmetry is transcribed,
    not corrected.

    Parameters
    ----------
    r_stress_point :
        Radius at which the stress is wanted (m).
    r_cs_inner, r_cs_outer :
        CS inner and outer radii (m).
    j_cs :
        CS current density (A/m^2).
    b_cs_inner :
        Field at the CS inner radius (T).
    f_poisson_cs_structure :
        Poisson's ratio of the CS structure.

    Returns
    -------
    :
        Radial stress (Pa).
    """
    alpha = r_cs_outer / r_cs_inner
    epsilon = r_stress_point / r_cs_inner
    b_cs_outer = 0.0

    k = ((alpha * b_cs_inner - b_cs_outer) * j_cs * r_cs_inner) / (alpha - 1.0)
    m = ((b_cs_inner - b_cs_outer) * j_cs * r_cs_inner) / (alpha - 1.0)

    hp_term_1 = k * ((2.0 + f_poisson_cs_structure) / (3.0 * (alpha + 1.0)))
    hp_term_2 = (
        alpha**2 + alpha + 1.0 - (alpha**2 / epsilon**2) - epsilon * (alpha + 1.0)
    )
    hp_term_2 = jnp.where(jnp.abs(hp_term_2) <= _RADIAL_TERM_SNAP, 0.0, hp_term_2)

    hp_term_3 = m * ((3.0 + f_poisson_cs_structure) / 8.0)
    hp_term_4 = alpha**2 + 1.0 - alpha**2 / epsilon**2 - epsilon**2
    hp_term_4 = jnp.where(jnp.abs(hp_term_4) <= _RADIAL_TERM_SNAP, 0.0, hp_term_4)

    return hp_term_1 * hp_term_2 - hp_term_3 * hp_term_4


def calculate_cs_self_peak_midplane_axial_stress(
    r_cs_outer, dz_cs_half, c_cs_peak, a_cs_toroidal
):
    """Axial self-force at the CS midplane and the stress it produces.

    Ports `CSCoil.calculate_cs_self_peak_midplane_axial_stress`,
    `process/models/pfcoil.py:4051-4128` (Iwasa, *Case Studies in Superconducting
    Magnets*), with `scipy.special.ellipk`/`ellipe` replaced by `_ellipk`/`_ellipe` --
    see this module's docstring for why those are AGM here and A&S fits in `fields.py`.

    Parameters
    ----------
    r_cs_outer :
        CS outer radius (m).
    dz_cs_half :
        CS half-height (m).
    c_cs_peak :
        Peak CS coil current (A).
    a_cs_toroidal :
        CS top-down toroidal area (m^2).

    Returns
    -------
    tuple
        `(stress_z_cs_self_peak_midplane, forc_z_cs_self_peak_midplane)` -- the
        un-smeared axial stress (Pa) and the axial force (N).
    """
    kb2 = (4.0 * r_cs_outer**2) / (4.0 * r_cs_outer**2 + dz_cs_half**2)
    k2b2 = (4.0 * r_cs_outer**2) / (4.0 * r_cs_outer**2 + 4.0 * dz_cs_half**2)

    axial_term_1 = -(RMU0 / 2.0) * (c_cs_peak / (2.0 * dz_cs_half)) ** 2
    axial_term_2 = (
        2.0
        * dz_cs_half
        * jnp.sqrt(4.0 * r_cs_outer**2 + dz_cs_half**2)
        * (_ellipk(kb2) - _ellipe(kb2))
    )
    axial_term_3 = (
        2.0
        * dz_cs_half
        * jnp.sqrt(4.0 * r_cs_outer**2 + 4.0 * dz_cs_half**2)
        * (_ellipk(k2b2) - _ellipe(k2b2))
    )

    force = axial_term_1 * (axial_term_2 - axial_term_3)
    return force / (0.5 * a_cs_toroidal), force


def calculate_tresca_stress(stress_x, stress_y, stress_z):
    """Tresca (maximum shear) criterion from three principal stresses (Pa).

    Ports `process/models/engineering/materials.py:53-82`, unchanged.

    Parameters
    ----------
    stress_x, stress_y, stress_z :
        The three principal stresses (Pa).

    Returns
    -------
    :
        `max(|sx - sy|, |sy - sz|, |sx - sz|)` (Pa).
    """
    return jnp.maximum(
        jnp.maximum(jnp.abs(stress_x - stress_y), jnp.abs(stress_y - stress_z)),
        jnp.abs(stress_x - stress_z),
    )


def calculate_von_mises_stress(
    stress_x,
    stress_y,
    stress_z,
    stress_shear_xy,
    stress_shear_yz,
    stress_shear_zx,
):
    """Von Mises criterion from three principal stresses and three shears (Pa).

    Ports `process/models/engineering/materials.py:84-129`, unchanged. `ohcalc` passes
    all three shear components as literal `0.0` (`:3513-3521`); they are kept in the
    signature because the source function has them, and the node below supplies the
    zeros at the call site rather than folding them into the body.

    Parameters
    ----------
    stress_x, stress_y, stress_z :
        The three principal stresses (Pa).
    stress_shear_xy, stress_shear_yz, stress_shear_zx :
        The three shear stresses (Pa).

    Returns
    -------
    :
        Von Mises stress (Pa).
    """
    return jnp.sqrt(
        0.5
        * (
            (stress_x - stress_y) ** 2
            + (stress_y - stress_z) ** 2
            + (stress_z - stress_x) ** 2
            + 6 * (stress_shear_xy**2 + stress_shear_yz**2 + stress_shear_zx**2)
        )
    )


def calculate_cs_stresses(
    r_cs_inner,
    r_cs_outer,
    r_cs_middle,
    dz_cs_full,
    a_cs_toroidal,
    j_cs_pulse_start,
    b_cs_peak_pulse_start,
    c_cs_peak_ma,
    f_poisson_cs_structure,
    f_a_cs_turn_steel,
):
    """The CS's whole stress state at the beginning of pulse.

    Ports `ohcalc`'s superconducting stress block (`process/models/pfcoil.py:3398-3521`)
    as one function, because the six outputs are six readings of one state and PROCESS
    computes them in one straight line with no branch between them.

    **Every stress is evaluated at the beginning of pulse**, not at the end of flat-top:
    all five calls take `j_cs = j_cs_pulse_start` and `b_cs_inner =
    b_cs_peak_pulse_start` (`:3406-3419`, `:3467-3474`, `:3476-3484`). The CS is at its
    hardest-worked at BOP, and this is where the `b_cs_peak_pulse_start` half of
    `fields.CSCoilPeakField` is actually consumed.

    **Three different radii, and each is PROCESS's own choice**: the hoop stress is
    taken at the *inner* radius, the peak radial stress at the *mean* radius, and the
    "inner" radial stress at the inner radius again (where it is identically zero, see
    `_RADIAL_TERM_SNAP`). The axial stress has no radius -- it is a midplane force
    divided by the toroidal area.

    Parameters
    ----------
    r_cs_inner, r_cs_outer, r_cs_middle :
        CS inner, outer and mean radii (m).
    dz_cs_full :
        CS full height (m); the axial stress takes half of it.
    a_cs_toroidal :
        CS top-down toroidal area (m^2).
    j_cs_pulse_start :
        CS current density at the beginning of pulse (A/m^2).
    b_cs_peak_pulse_start :
        Peak CS field at the beginning of pulse (T).
    c_cs_peak_ma :
        Peak CS coil current (MA) -- `.pf_coil.c_pf_cs_coils_peak_ma[6]`. Converted to
        amperes here, as `ohcalc:3428` does at the call site.
    f_poisson_cs_structure :
        Poisson's ratio of the CS structure. `.tfcoil.poisson_steel`.
    f_a_cs_turn_steel :
        Steel area fraction of the CS turn cross-section.

    Returns
    -------
    tuple
        `(stress_hoop_cs_inner, stress_z_cs_self_peak_midplane,
        forc_z_cs_self_peak_midplane, stress_radial_cs_peak, stress_radial_cs_inner,
        stress_shear_cs_peak, stress_mises_cs_peak)`, all in Pa except the force (N).
    """
    stress_hoop_cs_inner = calculate_cs_hoop_stress(
        r_stress_point=r_cs_inner,
        r_cs_inner=r_cs_inner,
        r_cs_outer=r_cs_outer,
        j_cs=j_cs_pulse_start,
        b_cs_inner=b_cs_peak_pulse_start,
        f_poisson_cs_structure=f_poisson_cs_structure,
        f_a_cs_turn_steel=f_a_cs_turn_steel,
    )

    stress_z, force_z = calculate_cs_self_peak_midplane_axial_stress(
        r_cs_outer=r_cs_outer,
        dz_cs_half=dz_cs_full / 2.0,
        c_cs_peak=c_cs_peak_ma * 1.0e6,
        a_cs_toroidal=a_cs_toroidal,
    )

    stress_radial_cs_peak = calculate_cs_radial_stress(
        r_stress_point=r_cs_middle,
        r_cs_inner=r_cs_inner,
        r_cs_outer=r_cs_outer,
        j_cs=j_cs_pulse_start,
        b_cs_inner=b_cs_peak_pulse_start,
        f_poisson_cs_structure=f_poisson_cs_structure,
    )
    stress_radial_cs_inner = calculate_cs_radial_stress(
        r_stress_point=r_cs_inner,
        r_cs_inner=r_cs_inner,
        r_cs_outer=r_cs_outer,
        j_cs=j_cs_pulse_start,
        b_cs_inner=b_cs_peak_pulse_start,
        f_poisson_cs_structure=f_poisson_cs_structure,
    )

    return (
        stress_hoop_cs_inner,
        stress_z,
        force_z,
        stress_radial_cs_peak,
        stress_radial_cs_inner,
        calculate_tresca_stress(
            stress_x=stress_hoop_cs_inner,
            stress_y=stress_z,
            stress_z=stress_radial_cs_peak,
        ),
        calculate_von_mises_stress(
            stress_x=stress_hoop_cs_inner,
            stress_y=stress_z,
            stress_z=stress_radial_cs_peak,
            stress_shear_xy=0.0,
            stress_shear_yz=0.0,
            stress_shear_zx=0.0,
        ),
    )


class CSCoilStresses(ExplicitFunction):
    """cottax node: `.tokamak.cs_coil.stresses`.

    Owns the six stress fields of `ohcalc`'s superconducting arm plus the axial force.
    Added 2026-08-27 for `optimise_design.md` §11.5's constraint-72 row:
    `.pf_coil.stress_shear_cs_peak` was a boundary zero against PROCESS's converged
    `1.1647e9` Pa.

    **Constraint 72 does not clear with this node, and cannot from this side.** It reads
    two variables -- `.pf_coil.stress_shear_cs_peak`, owned here, and
    `.tfcoil.sig_tf_cs_bucked`, which the TF stress block owns and which is `None` even
    in PROCESS's own converged `DataStructure` (it is never written at
    `i_tf_bucking = 1`; §11.5 records that separately). So this closes the half that can
    be closed.

    Occupant for `i_pf_conductor = SUPERCONDUCTING`, the package's single supported arm
    -- PROCESS's `else` (`:3532-3538`) sets the steel area to zero and computes no
    stresses at all, which would be a node owning nothing.
    """

    stress_hoop_cs_inner = OutputInto(pf_coil)
    stress_z_cs_self_peak_midplane = OutputInto(pf_coil)
    forc_z_cs_self_peak_midplane = OutputInto(pf_coil)
    stress_radial_cs_peak = OutputInto(pf_coil)
    stress_radial_cs_inner = OutputInto(pf_coil)
    stress_shear_cs_peak = OutputInto(pf_coil)
    stress_mises_cs_peak = OutputInto(pf_coil)

    def __call__(
        self,
        r_cs_inner=From(pf_coil),
        r_cs_outer=From(pf_coil),
        r_cs_middle=From(pf_coil),
        dz_cs_full=From(pf_coil),
        a_cs_toroidal=From(pf_coil),
        j_cs_pulse_start=From(pf_coil),
        b_cs_peak_pulse_start=From(pf_coil),
        c_pf_cs_coils_peak_ma=From(pf_coil),
        poisson_steel=From(tfcoil),
        f_a_cs_turn_steel=From(pf_coil),
    ):
        return calculate_cs_stresses(
            r_cs_inner=r_cs_inner,
            r_cs_outer=r_cs_outer,
            r_cs_middle=r_cs_middle,
            dz_cs_full=dz_cs_full,
            a_cs_toroidal=a_cs_toroidal,
            j_cs_pulse_start=j_cs_pulse_start,
            b_cs_peak_pulse_start=b_cs_peak_pulse_start,
            c_cs_peak_ma=c_pf_cs_coils_peak_ma[CS_INDEX],
            f_poisson_cs_structure=poisson_steel,
            f_a_cs_turn_steel=f_a_cs_turn_steel,
        )
