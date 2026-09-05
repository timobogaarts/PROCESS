"""Harness cases for the ported stellarator geometry chunk (1C).

Follows `test_build.py`/`test_structure.py`'s shape: no test functions
here, just reference adapters, the ports, and sample points, subclassing the tier the
audit record assigns.

`st_new_config` starts with `load_stellarator_config(istell, ...)`, which this chunk's
audit record deliberately keeps out of the traced port (device-config *table selection*
is graph-assembly-time setup, not a computation -- see `preset_config.md`, unit #8, and
`geometry.py`'s module docstring). To exercise `st_new_config` itself
without that call clobbering the `stella_config_*` fields a sample sets, the reference
adapters below patch `load_stellarator_config` out to a no-op for the duration of the
call -- the same "reference calls PROCESS through the port's own signature" contract as
every other unit, just with PROCESS's own non-traced preamble stubbed rather than
threaded through as arguments.
"""

from unittest.mock import patch

import numpy as np

import process.models.stellarator.stellarator as stellarator_module
from functional_process._harness import Tier1Contract, legacy_sample
from functional_process.cottax.stellarator.geometry import (
    calculate_default_aspect_ratio,
    calculate_stellarator_plasma_geometry,
    calculate_stellarator_scaling_factors,
)
from process.core.model import DataStructure
from process.models.stellarator.stellarator import Stellarator


def _stellarator():
    """A `Stellarator` instance whose sub-models are never called by this chunk."""
    stellarator = Stellarator(*([None] * 12))
    stellarator.data = DataStructure()
    return stellarator


def _baseline_data():
    """A `DataStructure` with every field `st_new_config` reads set to a sane value.

    `.numerics.ixc` defaults to an all-zero array (`1 not in ixc` is `True` out of the
    box), so the baseline leaves `aspect` config-owned unless a caller overrides it --
    matching whichever of the two nodes below the caller wants to exercise.
    """
    data = DataStructure()
    data.numerics.ixc = np.array([0])
    data.globals.output_prefix = ""
    data.physics.rmajor = 22.0
    data.physics.aspect = 12.33
    data.physics.b_plasma_toroidal_on_axis = 5.6
    data.stellarator.f_st_coil_aspect = 1.0
    data.stellarator_config.stella_config_coilspermodule = 10
    data.stellarator_config.stella_config_symmetry = 5
    data.stellarator_config.stella_config_rmajor_ref = 22.2
    data.stellarator_config.stella_config_rminor_ref = 1.8
    data.stellarator_config.stella_config_aspect_ref = 12.33
    data.stellarator_config.stella_config_bt_ref = 5.6
    data.stellarator_config.stella_config_coil_rmajor = 22.44
    data.stellarator_config.stella_config_coil_rminor = 4.76
    data.stellarator_config.stella_config_min_plasma_coil_distance = 1.9
    return data


def _run_st_new_config(data):
    """Run `st_new_config` with `load_stellarator_config` stubbed to a no-op.

    See module docstring: the config-loading call is deliberately out of scope for this
    port, and would otherwise clobber whatever `stella_config_*` values a sample set.
    """
    stellarator = _stellarator()
    stellarator.data = data
    with patch.object(
        stellarator_module, "load_stellarator_config", lambda *a, **k: None
    ):
        stellarator.st_new_config()
    return data


def _reference_default_aspect_ratio(stella_config_aspect_ref):
    """Call PROCESS's `st_new_config` (`1 not in ixc` branch) through the port's signature."""
    data = _baseline_data()
    data.numerics.ixc = np.array([0])  # aspect is NOT an active iteration variable
    data.stellarator_config.stella_config_aspect_ref = stella_config_aspect_ref

    _run_st_new_config(data)
    return data.physics.aspect


def _reference_stellarator_scaling_factors(
    rmajor,
    aspect,
    b_plasma_toroidal_on_axis,
    f_st_coil_aspect,
    stella_config_coilspermodule,
    stella_config_symmetry,
    stella_config_rmajor_ref,
    stella_config_rminor_ref,
    stella_config_aspect_ref,
    stella_config_bt_ref,
    stella_config_coil_rmajor,
    stella_config_coil_rminor,
    stella_config_min_plasma_coil_distance,
):
    """Call PROCESS's `st_new_config` (`1 in ixc` branch -- `aspect` externally supplied)
    through the port's signature."""
    data = _baseline_data()
    data.numerics.ixc = np.array([1])  # aspect IS an active iteration variable
    data.physics.rmajor = rmajor
    data.physics.aspect = aspect
    data.physics.b_plasma_toroidal_on_axis = b_plasma_toroidal_on_axis
    data.stellarator.f_st_coil_aspect = f_st_coil_aspect
    data.stellarator_config.stella_config_coilspermodule = stella_config_coilspermodule
    data.stellarator_config.stella_config_symmetry = stella_config_symmetry
    data.stellarator_config.stella_config_rmajor_ref = stella_config_rmajor_ref
    data.stellarator_config.stella_config_rminor_ref = stella_config_rminor_ref
    data.stellarator_config.stella_config_aspect_ref = stella_config_aspect_ref
    data.stellarator_config.stella_config_bt_ref = stella_config_bt_ref
    data.stellarator_config.stella_config_coil_rmajor = stella_config_coil_rmajor
    data.stellarator_config.stella_config_coil_rminor = stella_config_coil_rminor
    data.stellarator_config.stella_config_min_plasma_coil_distance = (
        stella_config_min_plasma_coil_distance
    )

    _run_st_new_config(data)
    return (
        data.physics.rminor,
        data.physics.eps,
        data.tfcoil.n_tf_coils,
        data.stellarator.f_st_rmajor,
        data.stellarator.f_st_rminor,
        data.stellarator.f_st_aspect,
        data.stellarator.f_st_n_coils,
        data.stellarator.f_st_b,
        data.stellarator.r_coil_major,
        data.stellarator.r_coil_minor,
        data.stellarator.f_coil_shape,
    )


def _reference_stellarator_plasma_geometry(
    f_st_rmajor,
    f_st_rminor,
    rminor,
    stella_config_vol_plasma,
    stella_config_plasma_surface,
):
    """Call PROCESS's `st_geom` through the port's signature. No switches, no calls out."""
    stellarator = _stellarator()
    data = stellarator.data
    data.stellarator.f_st_rmajor = f_st_rmajor
    data.stellarator.f_st_rminor = f_st_rminor
    data.physics.rminor = rminor
    data.stellarator_config.stella_config_vol_plasma = stella_config_vol_plasma
    data.stellarator_config.stella_config_plasma_surface = stella_config_plasma_surface

    stellarator.st_geom()
    return (
        data.physics.vol_plasma,
        data.physics.a_plasma_surface,
        data.physics.a_plasma_poloidal,
        data.physics.a_plasma_surface_outboard,
    )


class TestDefaultAspectRatio(Tier1Contract):
    """`st_new_config` (`1 not in ixc` branch) -> `calculate_default_aspect_ratio`."""

    audit_record = "models/stellarator/geometry.md"
    reference = _reference_default_aspect_ratio
    ported = calculate_default_aspect_ratio

    samples = [
        # HELIAS5B's reference aspect ratio (`preset_config.py`'s `HELIAS5B["aspect_ref"]`).
        legacy_sample("default-aspect-helias5b", stella_config_aspect_ref=12.33),
    ]

    fuzz_bounds = {
        "stella_config_aspect_ref": (1.5, 30.0),
    }


class TestStellaratorScalingFactors(Tier1Contract):
    """`st_new_config` (unconditional body) -> `calculate_stellarator_scaling_factors`."""

    audit_record = "models/stellarator/geometry.md"
    reference = _reference_stellarator_scaling_factors
    ported = calculate_stellarator_scaling_factors

    # tests/unit/models/stellarator/test_stellarator.py::test_stgeom's `stgeomparam` is
    # generated from helias_5b.IN.DAT and gives `f_st_rmajor`/`f_st_rminor` at
    # rmajor=22/rminor=1.7842660178426601 -- consistent with HELIAS5B's
    # rmajor_ref=22.2, aspect_ref=12.33 (22/22.2 = 0.990990990..., matching
    # `f_st_rmajor` there). No PROCESS unit test covers `st_new_config` directly, so
    # this legacy point is reconstructed from HELIAS5B's own table
    # (`preset_config.py`) plus that consistent geometry rather than lifted whole.
    samples = [
        legacy_sample(
            "scaling-helias5b",
            rmajor=22.0,
            aspect=12.33,
            b_plasma_toroidal_on_axis=5.6,
            f_st_coil_aspect=1.0,
            stella_config_coilspermodule=10,
            stella_config_symmetry=5,
            stella_config_rmajor_ref=22.2,
            stella_config_rminor_ref=1.80,
            stella_config_aspect_ref=12.33,
            stella_config_bt_ref=5.6,
            stella_config_coil_rmajor=22.44,
            stella_config_coil_rminor=4.76,
            stella_config_min_plasma_coil_distance=1.9,
        ),
    ]

    fuzz_bounds = {
        "rmajor": (1.0, 30.0),
        "aspect": (1.5, 30.0),
        "b_plasma_toroidal_on_axis": (1.0, 20.0),
        "f_st_coil_aspect": (0.5, 3.0),
        "stella_config_coilspermodule": (1.0, 20.0),
        "stella_config_symmetry": (1.0, 10.0),
        "stella_config_rmajor_ref": (1.0, 30.0),
        "stella_config_rminor_ref": (0.1, 5.0),
        "stella_config_aspect_ref": (1.5, 30.0),
        "stella_config_bt_ref": (1.0, 20.0),
        "stella_config_coil_rmajor": (1.0, 30.0),
        "stella_config_coil_rminor": (0.1, 10.0),
        "stella_config_min_plasma_coil_distance": (0.1, 5.0),
    }


class TestStellaratorPlasmaGeometry(Tier1Contract):
    """`st_geom` -> `calculate_stellarator_plasma_geometry`."""

    audit_record = "models/stellarator/geometry.md"
    reference = _reference_stellarator_plasma_geometry
    ported = calculate_stellarator_plasma_geometry

    # tests/unit/models/stellarator/test_stellarator.py::test_stgeom, generated from
    # helias_5b.IN.DAT (`StgeomParam`'s first case). `rmajor` is set by that test too,
    # but `st_geom` never reads `.physics.rmajor` -- only `.physics.rminor` -- so it is
    # not part of this port's signature and is omitted here.
    samples = [
        legacy_sample(
            "stgeom-helias5b",
            f_st_rmajor=0.99099099099099097,
            f_st_rminor=0.99125889880147788,
            rminor=1.7842660178426601,
            stella_config_vol_plasma=1422.6300000000001,
            stella_config_plasma_surface=1960,
        ),
    ]

    fuzz_bounds = {
        "f_st_rmajor": (0.1, 3.0),
        "f_st_rminor": (0.1, 3.0),
        "rminor": (0.1, 10.0),
        "stella_config_vol_plasma": (1.0, 5000.0),
        "stella_config_plasma_surface": (1.0, 5000.0),
    }
