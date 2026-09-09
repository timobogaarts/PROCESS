"""The Central Solenoid's stress state -- `ohcalc`'s superconducting-coil stress block.
"""

from cottax.interfaces.pytree_namespace_module import (
    ExplicitFunction,
    From,
    OutputInto,
)

from functional_process.cottax.paths import pf_coil, tfcoil
from functional_process.models.pfcoil.stresses import (
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
    """cottax node: `.tokamak.cs_coil.stresses`."""

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
