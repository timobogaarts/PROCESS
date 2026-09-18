"""Harness cases for the ported stellarator structural masses (chunk 1D).

Follows `test_density_limits.py`'s shape: no test functions here, just the reference
adapter, the port, and the sample points, subclassing the tier the audit record assigns.

Neither `st_strc` nor `sc_tf_coil_nuclear_heating_iter90` (see the sibling 1F test module)
calls into another model, so the reference adapter only needs a bare `Stellarator`
instance with `.data` attached -- the twelve injected sub-models in its constructor are
never touched by either method, so `None` stands in for all of them.
"""

from functional_process.tests._harness import Tier1Contract
from functional_process.tests._harness.process_reference import process_reference
from functional_process.tests._harness.sample_store import FROM_FILE
from functional_process.cottax.models.stellarator.structure import (
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


_reference_structure_masses = process_reference(
    _stellarator,
    "st_strc",
    ("structure.aintmass", "structure.clgsmass", "structure.coldmass"),
    call_args=(False,),
)


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
    samples = FROM_FILE

    fuzz = True


class TestIntercoilMassScalingReference(Tier1Contract):
    """`st_strc`'s `msupstr` -> `calculate_intercoil_mass_scaling_reference`."""

    audit_record = "models/stellarator/structure.md"
    reference = _reference_intercoil_mass_scaling_reference
    ported = calculate_intercoil_mass_scaling_reference

    samples = FROM_FILE

    fuzz = True
