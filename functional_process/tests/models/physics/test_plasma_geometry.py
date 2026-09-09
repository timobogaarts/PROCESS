"""Harness cases for the ported plasma geometry model (registry unit #24).

`plasma_angles_arcs`, `plasma_poloidal_perimeter`, `plasma_surface_area`,
`plasma_volume`, `plasma_cross_section` and `sauter_geometry` are diffed directly
against `PlasmaGeom`'s own `@staticmethod`s -- they take no `self.data` access at all,
so no adapter is needed and `reference = staticmethod(PlasmaGeom.<name>)` is exact.

`calculate_minor_radius`, `calculate_shape_ipdg89_x_point`,
`calculate_shape_create_data_eu_demo_x_point` and `calculate_geometry_double_arc` have
no PROCESS function of the same shape (`run()`'s preamble, two `i_plasma_geometry`
branches and the double-arc arm are each inline code, not standalone methods), so each
is diffed against `_make_plasma_geom`, a factory that builds a real `DataStructure`
with `large_tokamak_eval.IN.DAT`'s own switch values, calls the real, bound
`PlasmaGeom.run()`, and reads the relevant fields back -- the "close the `data`
backdoor" technique used throughout this harness, and the strongest oracle available
here: it validates against the actual stateful method, not a transcription of it.

Legacy sample values for `plasma_angles_arcs`/`plasma_surface_area`/`plasma_volume`/
`plasma_cross_section` are lifted from `tests/unit/models/physics/test_plasma_geom.py`.
`plasma_poloidal_perimeter` and `sauter_geometry` have no legacy point there (that file
never exercises them with recorded expectations for this port's purposes), so their
samples are physically-plausible synthetic points instead; the harness computes
"expected" by calling the reference itself; the `Sample` never carries a stored answer.

**Domain guard, per `plasma_geometry.md`'s open question 3 (D1).** `plasma_angles_arcs`
has no branch selection and returns silently wrong-signed geometry for
`kappa < 1 + triang`, with `ZeroDivisionError` exactly at the boundary. Every sample and
fuzz bound below is chosen to keep `kappa` comfortably above `1 + triang` (worst case in
`fuzz_bounds`: `kappa >= 1.6`, `triang <= 0.3`, so `kappa > 1 + triang` always by at
least `0.3`).
"""

import functools

from functional_process.cottax._harness import Tier1Contract
from functional_process.cottax._harness.process_reference import process_reference
from functional_process.cottax._harness.sample_store import FROM_FILE
from functional_process.cottax.models.physics.plasma_geometry import (
    calculate_geometry_double_arc,
    calculate_geometry_sauter,
    calculate_minor_radius,
    calculate_shape_create_data_eu_demo_x_point,
    calculate_shape_ipdg89_x_point,
    plasma_angles_arcs,
    plasma_cross_section,
    plasma_poloidal_perimeter,
    plasma_surface_area,
    plasma_volume,
    sauter_geometry,
)
from process.core.model import DataStructure
from process.models.physics.plasma_geometry import (
    PlasmaGeom,
    PlasmaGeometryModelType,
    PlasmaShapeModelType,
)


def _make_plasma_geom(
    i_plasma_geometry=PlasmaGeometryModelType.IPDG89_X_POINT, **physics
):
    """A real `PlasmaGeom`, `large_tokamak_eval.IN.DAT`'s switch values
    (`i_plasma_wall_gap=1`, `i_plasma_current=4`, `i_plasma_shape=0` so the double-arc
    arm of the compound Sauter switch is taken), plus whatever fixed physics fields a
    caller needs that aren't among the reference's own kwargs.
    """
    data = DataStructure()
    data.physics.i_plasma_geometry = i_plasma_geometry
    data.physics.i_plasma_wall_gap = 1
    data.physics.i_plasma_current = 4
    data.physics.i_plasma_shape = PlasmaShapeModelType.PROCESS_ORIGINAL
    for name, value in physics.items():
        setattr(data.physics, name, value)
    pg = PlasmaGeom()
    pg.data = data
    return pg


_reference_calculate_minor_radius = process_reference(
    functools.partial(_make_plasma_geom, kappa=1.7, triang=0.4),
    "run",
    ("rminor", "eps"),
)


_reference_calculate_shape_ipdg89_x_point = process_reference(
    functools.partial(_make_plasma_geom, rmajor=8.0, aspect=3.2),
    "run",
    ("kappa95", "triang95"),
)


# Switch values match `low_aspect_ratio_DEMO.IN.DAT` (the value-10 regression input):
# `i_plasma_geometry = 10` (`CREATE_DATA_EU_DEMO_X_POINT`, `:372`), `rmajor` held at the
# file's own `8.6` (`:164`) -- it feeds only `rminor`/`eps`, not this branch's three
# outputs. The input file's `kappa = 1.848` initial value is deliberately *not* set:
# value 10 overwrites `kappa`, which is exactly the ownership this occupant claims.
_reference_calculate_shape_create_data_eu_demo_x_point = process_reference(
    functools.partial(
        _make_plasma_geom,
        i_plasma_geometry=PlasmaGeometryModelType.CREATE_DATA_EU_DEMO_X_POINT,
        rmajor=8.6,
    ),
    "run",
    ("kappa95", "kappa", "triang95"),
)


def _reference_calculate_geometry_double_arc(
    rmajor, rminor, kappa, triang, f_vol_plasma
):
    """`aspect = rmajor / rminor` is folded in here: `calculate_geometry_double_arc`
    takes `rminor` directly and has no `aspect` kwarg of its own.
    """
    pg = _make_plasma_geom(
        rmajor=rmajor,
        aspect=rmajor / rminor,
        kappa=kappa,
        triang=triang,
        f_vol_plasma=f_vol_plasma,
    )
    pg.run()
    data = pg.data
    return (
        data.physics.len_plasma_poloidal,
        data.physics.vol_plasma,
        data.physics.a_plasma_poloidal,
        data.physics.a_plasma_surface,
    )


class TestPlasmaAnglesArcs(Tier1Contract):
    """`plasma_angles_arcs` -> the same, unchanged.

    Sample is `tests/unit/models/physics/test_plasma_geom.py::test_plasma_angles_arcs`'s
    point, verbatim -- genuinely legacy.
    """

    audit_record = "models/physics/plasma_geometry.md"
    reference = staticmethod(PlasmaGeom.plasma_angles_arcs)
    ported = plasma_angles_arcs

    samples = FROM_FILE

    fuzz = True


class TestPlasmaPoloidalPerimeter(Tier1Contract):
    """`plasma_poloidal_perimeter` -> the same, unchanged.

    No recorded legacy expectation in `tests/unit`; sample derived by running
    `plasma_angles_arcs` on the same operating point used elsewhere in this file, then
    diffed against `PlasmaGeom.plasma_poloidal_perimeter` at that point (still a real
    PROCESS reference call, just not a pre-recorded value).
    """

    audit_record = "models/physics/plasma_geometry.md"
    reference = staticmethod(PlasmaGeom.plasma_poloidal_perimeter)
    ported = plasma_poloidal_perimeter

    samples = FROM_FILE

    fuzz = True


class TestPlasmaSurfaceArea(Tier1Contract):
    """`plasma_surface_area` -> the same, unchanged.

    Sample is `tests/unit/models/physics/test_plasma_geom.py::test_plasma_surface_area`'s
    point, verbatim -- genuinely legacy.
    """

    audit_record = "models/physics/plasma_geometry.md"
    reference = staticmethod(PlasmaGeom.plasma_surface_area)
    ported = plasma_surface_area

    samples = FROM_FILE

    fuzz = True


class TestPlasmaVolume(Tier1Contract):
    """`plasma_volume` -> the same, unchanged.

    Sample is `tests/unit/models/physics/test_plasma_geom.py::test_plasma_volume`'s
    point, verbatim -- genuinely legacy.
    """

    audit_record = "models/physics/plasma_geometry.md"
    reference = staticmethod(PlasmaGeom.plasma_volume)
    ported = plasma_volume

    samples = FROM_FILE

    fuzz = True


class TestPlasmaCrossSection(Tier1Contract):
    """`plasma_cross_section` -> the same, unchanged.

    Sample is `tests/unit/models/physics/test_plasma_geom.py::
    test_plasma_cross_section`'s point, verbatim -- genuinely legacy.
    """

    audit_record = "models/physics/plasma_geometry.md"
    reference = staticmethod(PlasmaGeom.plasma_cross_section)
    ported = plasma_cross_section

    samples = FROM_FILE

    fuzz = True


class TestSauterGeometry(Tier1Contract):
    """`sauter_geometry` -> the same, unchanged.

    Ported for completeness (see the port module's docstring); no occupant class uses
    it yet, since the Sauter arm is not live on any input this pass covers. No recorded
    legacy expectation in `tests/unit`; sample is a plausible ITER-scale operating
    point, diffed against `PlasmaGeom.sauter_geometry` directly.
    """

    audit_record = "models/physics/plasma_geometry.md"
    reference = staticmethod(PlasmaGeom.sauter_geometry)
    ported = sauter_geometry

    samples = FROM_FILE

    fuzz = True


class TestCalculateGeometrySauter(Tier1Contract):
    """`calculate_geometry_sauter` -> reorders `sauter_geometry`'s own tuple.

    No PROCESS function of this exact shape (the reordering composition is new, see the
    port module's docstring); diffed against a thin adapter over `PlasmaGeom.
    sauter_geometry` rather than against `_run_plasma_geom`, since the Sauter arm is
    never taken by `_run_plasma_geom`'s own switch configuration.
    """

    audit_record = "models/physics/plasma_geometry.md"

    @staticmethod
    def reference(rmajor, rminor, kappa, triang, plasma_square):
        len_plasma_poloidal, a_plasma_surface, a_plasma_poloidal, vol_plasma = (
            PlasmaGeom.sauter_geometry(rminor, rmajor, kappa, triang, plasma_square)
        )
        return len_plasma_poloidal, vol_plasma, a_plasma_poloidal, a_plasma_surface

    ported = calculate_geometry_sauter

    samples = FROM_FILE

    fuzz = True


class TestCalculateMinorRadius(Tier1Contract):
    """`calculate_minor_radius` -> `PlasmaGeom.run()`'s unconditional preamble.

    No standalone PROCESS function of this shape; diffed against `_make_plasma_geom`,
    which calls the real, bound `run()` and reads `.physics.rminor`/`.physics.eps` back.
    """

    audit_record = "models/physics/plasma_geometry.md"
    reference = staticmethod(_reference_calculate_minor_radius)
    ported = calculate_minor_radius

    samples = FROM_FILE

    fuzz = True


class TestCalculateShapeIpdg89XPoint(Tier1Contract):
    """`calculate_shape_ipdg89_x_point` -> `i_plasma_geometry == IPDG89_X_POINT` (0).

    No standalone PROCESS function of this shape (it is 2 lines of `run()`'s dispatch);
    diffed against `_make_plasma_geom`, reading `.physics.kappa95`/`.physics.triang95`
    back after the real `run()` call.
    """

    audit_record = "models/physics/plasma_geometry.md"
    reference = staticmethod(_reference_calculate_shape_ipdg89_x_point)
    ported = calculate_shape_ipdg89_x_point

    samples = FROM_FILE

    fuzz = True


class TestCalculateShapeCreateDataEuDemoXPoint(Tier1Contract):
    """`calculate_shape_create_data_eu_demo_x_point` -> `i_plasma_geometry == 10`.

    No standalone PROCESS function of this shape (the branch is inline in `run()`'s
    dispatch, `plasma_geometry.py:362-397`); diffed against `_make_plasma_geom`, which
    calls the real, bound `run()` under `i_plasma_geometry = 10` and reads
    `.physics.kappa95`/`.kappa`/`.triang95` back.

    Two legacy points, one per arm of the branch's own `if kappa95 > 1.77:` (the C0-
    but-not-C1 corner fudge, audit record **D6**): the first is
    `low_aspect_ratio_DEMO.IN.DAT`'s own operating point (`aspect = 2.8`,
    `m_s_limit = 0.2`, `triang = 0.5` -> raw `kappa95 ~= 1.740`, fudge NOT taken --
    the live regression input never exercises the fudge), the second pushes
    `kappa95` above 1.77 (`aspect = 2.6`, `m_s_limit = 0.0` -> raw `kappa95 ~= 1.812`)
    so the `jnp.where`/`safe_pow` arm is value- and gradient-checked too.

    Fuzz bounds: `aspect` spans the fit's documented validity range (2.6-3.6, PROCESS
    issue #1648); outside it the radicand of the fit's square root can go negative
    (**F3**). `triang <= 0.5` keeps the reference `run()`'s downstream
    `plasma_angles_arcs` call inside D1's domain (`kappa > 1 + triang`: the fit gives
    `kappa >= 1.72` over this whole box, margin >= 0.22). The box straddles the 1.77
    kink deliberately -- the kink is a measure-zero set, and both arms should be
    sampled.
    """

    audit_record = "models/physics/plasma_geometry.md"
    reference = staticmethod(_reference_calculate_shape_create_data_eu_demo_x_point)
    ported = calculate_shape_create_data_eu_demo_x_point

    samples = FROM_FILE

    fuzz = True


class TestCalculateGeometryDoubleArc(Tier1Contract):
    """`calculate_geometry_double_arc` -> the geometry-model arm's double-arc arm.

    No standalone PROCESS function of this shape; diffed against `_make_plasma_geom`
    (whose switch configuration always takes the double-arc arm), reading
    `.physics.len_plasma_poloidal`/`.vol_plasma`/`.a_plasma_poloidal`/`.a_plasma_surface`
    back after the real `run()` call. This is the node that owns three of the slot's
    five target outputs (`a_plasma_poloidal`, `a_plasma_surface`, `vol_plasma`).
    """

    audit_record = "models/physics/plasma_geometry.md"
    reference = staticmethod(_reference_calculate_geometry_double_arc)
    ported = calculate_geometry_double_arc

    samples = FROM_FILE

    fuzz = True
