"""Harness cases for the ported stellarator structural masses (chunk 1D).

Follows `test_density_limits.py`'s shape: no test functions here, just the reference
adapter, the port, and the sample points, subclassing the tier the audit record assigns.

Neither `st_strc` nor `sc_tf_coil_nuclear_heating_iter90` (see the sibling 1F test module)
calls into another model, so the reference adapter only needs a bare `Stellarator`
instance with `.data` attached -- the twelve injected sub-models in its constructor are
never touched by either method, so `None` stands in for all of them.
"""

from functional_process._harness import Tier1Contract, legacy_sample
from functional_process.cottax.stellarator.structure import (
    calculate_intercoil_mass_scaling_reference,
    calculate_structure_masses,
)
from process.core.model import DataStructure
from process.models.stellarator.stellarator import Stellarator


def _stellarator():
    """A `Stellarator` instance whose sub-models are never called by `st_strc`."""
    stellarator = Stellarator(*([None] * 12))
    stellarator.data = DataStructure()
    return stellarator


def _reference_structure_masses(
    stella_config_coilsurface,
    f_st_rmajor,
    r_coil_minor,
    stella_config_coil_rminor,
    dx_tf_inboard_out_toroidal,
    len_tf_coil,
    n_tf_coils,
    b_plasma_toroidal_on_axis,
    den_steel,
    m_tf_coils_total,
    dewmkg,
):
    """Call PROCESS's `Stellarator.st_strc` through the port's signature."""
    stellarator = _stellarator()
    data = stellarator.data
    data.stellarator_config.stella_config_coilsurface = stella_config_coilsurface
    data.stellarator_config.stella_config_coil_rminor = stella_config_coil_rminor
    data.stellarator.f_st_rmajor = f_st_rmajor
    data.stellarator.r_coil_minor = r_coil_minor
    data.tfcoil.dx_tf_inboard_out_toroidal = dx_tf_inboard_out_toroidal
    data.tfcoil.len_tf_coil = len_tf_coil
    data.tfcoil.n_tf_coils = n_tf_coils
    data.physics.b_plasma_toroidal_on_axis = b_plasma_toroidal_on_axis
    data.fwbs.den_steel = den_steel
    data.tfcoil.m_tf_coils_total = m_tf_coils_total
    data.fwbs.dewmkg = dewmkg

    stellarator.st_strc(output=False)
    return data.structure.aintmass, data.structure.clgsmass, data.structure.coldmass


def _reference_intercoil_mass_scaling_reference(e_tf_magnetic_stored_total_gj):
    """`msupstr` is never stored to `data` -- reproduce the one-line formula directly.

    `st_strc` prints `msupstr` (`po.ovarre(..., "(empiricalmass)", msupstr)`) but never
    assigns it to any `data` field, so there is no `data`-mediated way to call it through
    `Stellarator`. This is exactly PROCESS's own formula, copied rather than re-derived,
    to keep the reference and the port textually independent.
    """
    m_struc = 1.3483e0 * (1000.0e0 * e_tf_magnetic_stored_total_gj) ** 0.7821e0
    return 1000.0e0 * m_struc


class TestStructureMasses(Tier1Contract):
    """`Stellarator.st_strc` -> `calculate_structure_masses`."""

    audit_record = "models/stellarator/structure.md"
    reference = _reference_structure_masses
    ported = calculate_structure_masses

    # tests/unit/models/stellarator/test_stellarator.py::test_ststrc, generated from
    # helias_5b.IN.DAT. The source test sets `r_coil_minor` from `f_st_rmajor`'s value
    # (not its own field) -- a quirk of that auto-generated test, reproduced here rather
    # than corrected, since this sample exists to match a known-good PROCESS run.
    # `b_plasma_toroidal_on_axis` isn't overridden by that test either; its default
    # (`process/data_structure/physics_variables.py`, 5.68) is used here explicitly.
    samples = [
        legacy_sample(
            "ststrc-helias5b-dewmkg0",
            stella_config_coilsurface=4817.6999999999998,
            f_st_rmajor=0.99099099099099097,
            r_coil_minor=0.99099099099099097,
            stella_config_coil_rminor=1.0,
            dx_tf_inboard_out_toroidal=0.67648706726464258,
            len_tf_coil=1664.8648648648648,
            n_tf_coils=1,
            b_plasma_toroidal_on_axis=5.68,
            den_steel=7800,
            m_tf_coils_total=5204872.8206625767,
            dewmkg=0,
        ),
        legacy_sample(
            "ststrc-helias5b-dewmkg-nonzero",
            stella_config_coilsurface=4817.6999999999998,
            f_st_rmajor=0.99099099099099097,
            r_coil_minor=0.99099099099099097,
            stella_config_coil_rminor=1.0,
            dx_tf_inboard_out_toroidal=0.67648706726464258,
            len_tf_coil=1664.8648648648648,
            n_tf_coils=1,
            b_plasma_toroidal_on_axis=5.68,
            den_steel=7800,
            m_tf_coils_total=5204872.8206625767,
            dewmkg=22397931.480129492,
        ),
    ]

    fuzz_bounds = {
        "stella_config_coilsurface": (100.0, 1.0e4),
        "f_st_rmajor": (0.5, 1.5),
        "r_coil_minor": (0.1, 5.0),
        "stella_config_coil_rminor": (0.1, 5.0),
        "dx_tf_inboard_out_toroidal": (0.01, 5.0),
        "len_tf_coil": (10.0, 5000.0),
        "n_tf_coils": (1.0, 100.0),
        "b_plasma_toroidal_on_axis": (1.0, 20.0),
        "den_steel": (1000.0, 10000.0),
        "m_tf_coils_total": (1.0e3, 1.0e8),
        "dewmkg": (0.0, 1.0e8),
    }


class TestIntercoilMassScalingReference(Tier1Contract):
    """`st_strc`'s `msupstr` -> `calculate_intercoil_mass_scaling_reference`."""

    audit_record = "models/stellarator/structure.md"
    reference = _reference_intercoil_mass_scaling_reference
    ported = calculate_intercoil_mass_scaling_reference

    samples = [
        # Same operating point as TestStructureMasses's legacy samples
        # (helias_5b.IN.DAT), for `e_tf_magnetic_stored_total_gj`.
        legacy_sample(
            "msupstr-helias5b",
            e_tf_magnetic_stored_total_gj=132.55990646265246,
        ),
    ]

    fuzz_bounds = {
        "e_tf_magnetic_stored_total_gj": (1.0, 1.0e4),
    }
