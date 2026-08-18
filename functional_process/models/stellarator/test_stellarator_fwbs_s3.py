"""Harness cases for `st_fwbs`'s S3 fragment (`stellarator.py:1030-1043`).

Audit record: `functional_process/models/stellarator/stellarator_fwbs_s3.md`.

`st_fwbs` (`process/models/stellarator/stellarator.py:481-1682`) is one 1200-line method;
S3 is a 14-line fragment inside it, not a separately callable `Stellarator` method the
way `st_strc`/`sc_tf_coil_nuclear_heating_iter90` are for the sibling `stellarator_D`/`_F`
units. Calling `st_fwbs` itself to exercise just S3 would also run S1/S2 first (many more
inputs, a `blktmodel == 1` arm that calls into not-fully-wired `hcpb.py` staticmethods
with a known `TypeError` bug on record -- `unit_registry.md` row 13), which is out of
this unit's scope and would make the test depend on machinery this audit deliberately
does not touch. Per the same precedent as `stellarator_D_structure.md`'s `msupstr`
(`test_stellarator_D_structure.py::_reference_intercoil_mass_scaling_reference`), the
reference below reproduces the fragment's one formula directly from source rather than
invoking `st_fwbs`, textually independent of the port in `stellarator_fwbs_s3.py`.

Like `divertor.md`'s `st_div`, no PROCESS unit test exercises this fragment directly
(`tests/unit/models/stellarator/test_stellarator.py` never asserts on `m_div_plate` or
`a_div_surface_total`), so there is no `legacy_sample` here -- coverage is fuzz-only.
"""

from functional_process._harness import Tier1Contract
from functional_process.models.stellarator.stellarator_fwbs_s3 import (
    calculate_divertor_plate_mass,
)


def _reference_divertor_plate_mass(
    a_div_surface_total,
    den_div_structure,
    f_vol_div_coolant,
    dx_div_plate,
):
    """`stellarator.py:1038-1043`'s `m_div_plate` formula, copied not re-derived.

    `first_call_stfwbs`'s conditional overwrite of `a_div_surface_total` (the `50.0e0`
    bootstrap) is not part of this formula -- see the audit record's "Framing" section --
    so `a_div_surface_total` is taken here exactly as the port takes it, as a plain
    argument standing in for whichever value (bootstrap or `Divertor`'s previous-call
    output) a driver would have fed into it.
    """
    return (
        a_div_surface_total
        * den_div_structure
        * (1.0e0 - f_vol_div_coolant)
        * dx_div_plate
    )


class TestDivertorPlateMass(Tier1Contract):
    """`st_fwbs`'s S3 fragment -> `calculate_divertor_plate_mass`."""

    audit_record = "models/stellarator/stellarator_fwbs_s3.md"
    reference = _reference_divertor_plate_mass
    ported = calculate_divertor_plate_mass

    # No PROCESS unit test exercises this fragment -- see module docstring, so there is
    # no `legacy_sample` to add here. Coverage is fuzz-only, at plausible stellarator
    # divertor-geometry bounds. `a_div_surface_total`'s range covers both the `50.0e0`
    # bootstrap literal and plausible converged `Divertor` outputs (`divertor.py`'s own
    # `fuzz_bounds` implies areas from ~tens to ~thousands of m2 given its own inputs).
    fuzz_bounds = {
        "a_div_surface_total": (1.0, 2000.0),
        "den_div_structure": (1.0e3, 2.0e4),
        "f_vol_div_coolant": (0.0, 0.9),
        "dx_div_plate": (0.001, 0.5),
    }
