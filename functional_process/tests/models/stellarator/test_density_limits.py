"""Harness cases for the ported stellarator density limits.

The first real use of the harness, and the shape every later unit should copy: declare
the reference, the port, the points, and inherit the tier's checks. There are no test
functions in this file.

The `_reference_*` adapters are the interesting part. PROCESS's `st_sudo_density_limit`
takes a `DataStructure` and reads two fields off it; the adapter binds a bare
`DataStructure` with just those two fields set, which is what turns the audit record's
claim "these are the only two reads, and closing the back-door is safe" into something
the test suite would catch if it were wrong.
"""

from functional_process.cottax._harness import (
    Tier1Contract,
    bounds_from_iteration_variables,
)
from functional_process.cottax._harness.sample_store import FROM_FILE
from functional_process.cottax.stellarator.density_limits import (
    calculate_ecrh_density_limit,
    calculate_sudo_density_limit,
)
from process.core.exceptions import ProcessValueError
from process.core.model import DataStructure
from process.models.stellarator.density_limits import (
    st_d_limit_ecrh,
    st_sudo_density_limit,
)


def _reference_sudo_density_limit(
    b_plasma_toroidal_on_axis,
    p_plasma_loss_mw,
    rmajor,
    rminor,
    nd_plasma_electrons_vol_avg,
    nd_plasma_electron_line,
):
    """Call PROCESS's `st_sudo_density_limit` through the port's signature."""
    data = DataStructure()
    data.physics.nd_plasma_electrons_vol_avg = nd_plasma_electrons_vol_avg
    data.physics.nd_plasma_electron_line = nd_plasma_electron_line
    return st_sudo_density_limit(
        b_plasma_toroidal_on_axis=b_plasma_toroidal_on_axis,
        powht=p_plasma_loss_mw,
        rmajor=rmajor,
        rminor=rminor,
        data=data,
    )


def _reference_ecrh_density_limit(
    gyro_frequency_max,
    b_plasma_toroidal_on_axis,
    i_plasma_pedestal=0,
):
    """Call PROCESS's `st_d_limit_ecrh` through the port's signature."""
    return st_d_limit_ecrh(
        gyro_frequency_max=gyro_frequency_max,
        bt_input=b_plasma_toroidal_on_axis,
        i_plasma_pedestal=i_plasma_pedestal,
    )


class TestSudoDensityLimit(Tier1Contract):
    """`st_sudo_density_limit` -> `calculate_sudo_density_limit`."""

    audit_record = "models/stellarator/density_limits.md"
    reference = _reference_sudo_density_limit
    ported = calculate_sudo_density_limit

    reference_domain_errors = (ProcessValueError,)

    # The first two points are from test_stellarator.py::test_stdlim in PROCESS's own
    # suite, itself generated from helias_5b.IN.DAT — already-validated realistic
    # operating points, reused rather than re-derived.
    samples = FROM_FILE

    fuzz_bounds = {
        **bounds_from_iteration_variables(
            "b_plasma_toroidal_on_axis",
            "rmajor",
            "nd_plasma_electrons_vol_avg",
        ),
        # Not iteration variables, so no PROCESS-sanctioned range to borrow. `rminor`
        # is derived from `rmajor`/`aspect` rather than varied directly; the rest span
        # the operating points above by a decade or two either way.
        "rminor": (0.1, 10.0),
        "p_plasma_loss_mw": (1.0, 1.0e4),
        "nd_plasma_electron_line": (2.0e19, 1.0e21),
    }


class TestEcrhDensityLimit(Tier1Contract):
    """`st_d_limit_ecrh` -> `calculate_ecrh_density_limit`."""

    audit_record = "models/stellarator/density_limits.md"
    reference = _reference_ecrh_density_limit
    ported = calculate_ecrh_density_limit

    # A precondition, not a domain to sample: see the port's docstring.
    static_argnames = ("i_plasma_pedestal",)

    samples = FROM_FILE

    fuzz_bounds = {
        **bounds_from_iteration_variables("b_plasma_toroidal_on_axis"),
        # Spans both sides of the min() kink at bt = gyro_frequency_max*2*pi/1.76e11,
        # which for this range lands inside the field bounds above — deliberately, so
        # the sampler exercises both branches.
        "gyro_frequency_max": (1.0e11, 1.0e12),
    }
    fuzz_fixed = {"i_plasma_pedestal": 0}
