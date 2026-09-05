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

from cottax.interfaces.pytree_namespace_module import (
    ExplicitFunction,
    From,
    OutputInto,
)

from functional_process.paths import pf_coil, tfcoil
from functional_process.pfcoil.stresses import (
    _ellipe,  # noqa: F401 -- re-exported for tests
    _ellipk,  # noqa: F401 -- re-exported for tests
    calculate_cs_hoop_stress,  # noqa: F401 -- re-exported for tests
    calculate_cs_radial_stress,  # noqa: F401 -- re-exported for tests
    calculate_cs_self_peak_midplane_axial_stress,  # noqa: F401 -- re-exported for tests
    calculate_cs_stresses,  # noqa: F401 -- re-exported for tests
    calculate_cs_stresses_from_full_width_current,
    calculate_tresca_stress,  # noqa: F401 -- re-exported for tests
    calculate_von_mises_stress,  # noqa: F401 -- re-exported for tests
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
        return calculate_cs_stresses_from_full_width_current(
            r_cs_inner=r_cs_inner,
            r_cs_outer=r_cs_outer,
            r_cs_middle=r_cs_middle,
            dz_cs_full=dz_cs_full,
            a_cs_toroidal=a_cs_toroidal,
            j_cs_pulse_start=j_cs_pulse_start,
            b_cs_peak_pulse_start=b_cs_peak_pulse_start,
            c_pf_cs_coils_peak_ma=c_pf_cs_coils_peak_ma,
            f_poisson_cs_structure=poisson_steel,
            f_a_cs_turn_steel=f_a_cs_turn_steel,
        )
