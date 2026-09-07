"""Mutual and self inductances of the PF coils, the CS and the plasma."""

import equinox as eqx
from cottax.interfaces.pytree_namespace_module import ExplicitFunction, From, OutputInto

from functional_process.cottax.pfcoil import (
    SPHERICAL_TOKAMAK_TOPOLOGY,
    PFCoilTopology,
)
from functional_process.cottax.paths import build, pf_coil, physics
from functional_process.models.pfcoil.inductance import (
    NOH_PAD,  # noqa: F401 -- re-exported for tests/test_cold_start.py
    _cs_segments,  # noqa: F401 -- re-exported for tests/test_cold_start.py
    calculate_pf_cs_plasma_inductances,  # noqa: F401 -- re-exported for tests
    calculate_pf_cs_plasma_inductances_at_reference_width,
    calculate_pf_plasma_inductances_no_central_solenoid,  # noqa: F401 -- re-exported for tests
    calculate_pf_plasma_inductances_no_central_solenoid_for_topology,
    calculate_solenoid_self_inductance,  # noqa: F401 -- re-exported for tests
)

NPLAS = 1
"""`nplas`, a literal `1` in `induct` (`pfcoil.py:1734`) -- the plasma is one filament,
at `(rmajor, 0)`.
"""


class PFCoilInductance(ExplicitFunction):
    """cottax node: `.tokamak.pf_coil.inductance`."""

    ind_pf_cs_plasma_mutual = OutputInto(pf_coil)

    def __call__(
        self,
        rmajor=From(physics),
        ind_plasma=From(physics),
        dr_cs=From(build),
        r_cs_middle=From(pf_coil),
        r_pf_coil_middle=From(pf_coil),
        z_pf_coil_middle=From(pf_coil),
        r_pf_coil_inner=From(pf_coil),
        r_pf_coil_outer=From(pf_coil),
        z_pf_coil_upper=From(pf_coil),
        z_pf_coil_lower=From(pf_coil),
        n_pf_coil_turns=From(pf_coil),
    ):
        return calculate_pf_cs_plasma_inductances_at_reference_width(
            rmajor=rmajor,
            ind_plasma=ind_plasma,
            dr_cs=dr_cs,
            r_cs_middle=r_cs_middle,
            r_pf_coil_middle=r_pf_coil_middle,
            z_pf_coil_middle=z_pf_coil_middle,
            r_pf_coil_inner=r_pf_coil_inner,
            r_pf_coil_outer=r_pf_coil_outer,
            z_pf_coil_upper=z_pf_coil_upper,
            z_pf_coil_lower=z_pf_coil_lower,
            n_pf_coil_turns=n_pf_coil_turns,
        )


class PFCoilInductanceNoCentralSolenoid(ExplicitFunction):
    """cottax node: `.tokamak.pf_coil.inductance`, the `iohcl = 0` occupant."""

    topology: PFCoilTopology = eqx.field(static=True, default=SPHERICAL_TOKAMAK_TOPOLOGY)

    ind_pf_cs_plasma_mutual = OutputInto(pf_coil)

    def __call__(
        self,
        rmajor=From(physics),
        ind_plasma=From(physics),
        r_pf_coil_middle=From(pf_coil),
        z_pf_coil_middle=From(pf_coil),
        z_pf_coil_upper=From(pf_coil),
        z_pf_coil_lower=From(pf_coil),
        n_pf_coil_turns=From(pf_coil),
    ):
        return calculate_pf_plasma_inductances_no_central_solenoid_for_topology(
            rmajor=rmajor,
            ind_plasma=ind_plasma,
            r_pf_coil_middle=r_pf_coil_middle,
            z_pf_coil_middle=z_pf_coil_middle,
            z_pf_coil_upper=z_pf_coil_upper,
            z_pf_coil_lower=z_pf_coil_lower,
            n_pf_coil_turns=n_pf_coil_turns,
            topology=self.topology,
        )
