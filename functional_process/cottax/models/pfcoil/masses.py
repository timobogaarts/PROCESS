"""How big each coil has to be, and what it weighs."""

from cottax.interfaces import V

from functional_process.cottax.paths import fwbs, pf_coil, physics
from functional_process.models.pfcoil.masses import (
    calculate_pf_coil_masses,  # noqa: F401 -- re-exported for tests
    calculate_pf_coil_masses_from_elements,
    calculate_pf_coil_masses_no_central_solenoid,  # noqa: F401 -- re-exported for tests
    calculate_pf_coil_masses_no_central_solenoid_bound,
    calculate_pf_coil_masses_no_central_solenoid_for_topology,  # noqa: F401 -- re-exported for tests
    calculate_pf_coil_sizes,  # noqa: F401 -- re-exported for tests
    calculate_pf_coil_sizes_for_topology,  # noqa: F401 -- re-exported for tests
    calculate_pf_coil_sizes_from_elements,
    calculate_pf_coil_sizes_no_central_solenoid,
)

# cottax node: `.tokamak.pf_coil.sizes`.
#
# Which slot each coil occupies and whether there is a CS slot to fill is fixed by
# `fn` (`calculate_pf_coil_sizes_from_elements`, for `REFERENCE_TOPOLOGY`) rather than
# carried as a static field here -- see `PFCoilSizesNoCentralSolenoid` for the other
# arm.
PFCoilSizes = pf_coil.function(
    calculate_pf_coil_sizes_from_elements,
    "n_pf_coil_turns",
    "r_pf_coil_inner",
    "r_pf_coil_outer",
    "z_pf_coil_upper",
    "z_pf_coil_lower",
    "r_pf_coil_outer_max",
    reads={
        "rmajor": physics,
        "rminor": physics,
        "kappa": physics,
    },
)


# cottax node: `.tokamak.pf_coil.sizes`, the `iohcl = 0` occupant.
PFCoilSizesNoCentralSolenoid = pf_coil.function(
    calculate_pf_coil_sizes_no_central_solenoid,
    "n_pf_coil_turns",
    "r_pf_coil_inner",
    "r_pf_coil_outer",
    "z_pf_coil_upper",
    "z_pf_coil_lower",
    "r_pf_coil_outer_max",
    reads={
        "rmajor": physics,
        "rminor": physics,
        "kappa": physics,
    },
)


# cottax node: `.tokamak.pf_coil.masses`. The conductor densities are
# `.pf_coil.den_pf_conductor` / `.pf_coil.den_cs_conductor`, the entries of
# `.tfcoil.dcond` at the PF and CS superconductor indices, derived from a
# configuration's values (`cottax.input.native`).
PFCoilMasses = pf_coil.function(
    calculate_pf_coil_masses_from_elements,
    "m_pf_coil_conductor",
    "m_pf_coil_structure",
    "pfcaseth",
    "m_pf_coil_conductor_total",
    "m_pf_coil_structure_total",
    "m_pf_coil_max",
    "ricpf",
    "a_cs_steel_poloidal",
    "a_cs_cable_space",
    reads={
        "b_pf_coil_peak_0": V.pf_coil.b_pf_coil_peak[0],
        "b_pf_coil_peak_1": V.pf_coil.b_pf_coil_peak[1],
        "b_pf_coil_peak_2": V.pf_coil.b_pf_coil_peak[2],
        "b_pf_coil_peak_3": V.pf_coil.b_pf_coil_peak[3],
        "b_pf_coil_peak_4": V.pf_coil.b_pf_coil_peak[4],
        "b_pf_coil_peak_5": V.pf_coil.b_pf_coil_peak[5],
        "bpf2_0": V.pf_coil.bpf2[0],
        "bpf2_1": V.pf_coil.bpf2[1],
        "bpf2_2": V.pf_coil.bpf2[2],
        "bpf2_3": V.pf_coil.bpf2[3],
        "bpf2_4": V.pf_coil.bpf2[4],
        "bpf2_5": V.pf_coil.bpf2[5],
        "den_steel": fwbs,
    },
)


# cottax node: `.tokamak.pf_coil.masses`, the `iohcl = 0` occupant.
PFCoilMassesNoCentralSolenoid = pf_coil.function(
    calculate_pf_coil_masses_no_central_solenoid_bound,
    "m_pf_coil_conductor",
    "m_pf_coil_structure",
    "pfcaseth",
    "m_pf_coil_conductor_total",
    "m_pf_coil_structure_total",
    "m_pf_coil_max",
    "ricpf",
    reads={
        "den_steel": fwbs,
    },
)
