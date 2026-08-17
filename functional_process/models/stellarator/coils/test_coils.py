"""Harness cases for the ported subset of `coils/coils.py` (registry #10)."""

from functional_process._harness import Tier1Contract, legacy_sample
from functional_process.models.stellarator.coils.coils import (
    bmax_from_awp,
    j_crit_cable_from_fraction,
)
from process.core.model import DataStructure
from process.models.stellarator.coils.coils import (
    j_crit_cable_from_fraction as _process_j_crit_cable_from_fraction,
)


def _reference_bmax_from_awp(
    wp_width_radial,
    current,
    n_tf_coils,
    r_coil_major,
    r_coil_minor,
    stella_config_a1,
    stella_config_a2,
):
    """Call PROCESS's `bmax_from_awp` through the port's signature."""
    from process.models.stellarator.coils.coils import bmax_from_awp as _process_bmax

    data = DataStructure()
    data.stellarator_config.stella_config_a1 = stella_config_a1
    data.stellarator_config.stella_config_a2 = stella_config_a2
    return _process_bmax(
        wp_width_radial=wp_width_radial,
        current=current,
        n_tf_coils=n_tf_coils,
        r_coil_major=r_coil_major,
        r_coil_minor=r_coil_minor,
        data=data,
    )


class TestJCritCableFromFraction(Tier1Contract):
    """`j_crit_cable_from_fraction` -> itself (already pure in the source)."""

    audit_record = "models/stellarator/coils/coils.md"
    reference = staticmethod(_process_j_crit_cable_from_fraction)
    ported = j_crit_cable_from_fraction

    fuzz_bounds = {
        "j_crit_sc": (1.0, 1.0e4),
        "f_tf_conductor_copper": (0.0, 0.95),
        "f_he": (0.0, 0.95),
    }


class TestBmaxFromAwp(Tier1Contract):
    """`bmax_from_awp` -> `bmax_from_awp` (data back-door closed)."""

    audit_record = "models/stellarator/coils/coils.md"
    reference = _reference_bmax_from_awp
    ported = bmax_from_awp

    # tests/unit/models/stellarator/test_stellarator.py::test_bmax_from_awp.
    samples = [
        legacy_sample(
            "bmax-from-awp-helias",
            wp_width_radial=0.11792792792792792,
            current=12.711229086229087,
            n_tf_coils=50,
            r_coil_major=22.237837837837837,
            r_coil_minor=4.7171171171171169,
            stella_config_a1=0.688,
            stella_config_a2=0.025,
        ),
    ]

    fuzz_bounds = {
        "wp_width_radial": (0.01, 2.0),
        "current": (0.1, 100.0),
        "n_tf_coils": (10.0, 100.0),
        "r_coil_major": (5.0, 30.0),
        "r_coil_minor": (0.5, 6.0),
        "stella_config_a1": (0.1, 1.5),
        "stella_config_a2": (0.001, 0.1),
    }
