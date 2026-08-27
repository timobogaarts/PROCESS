"""Harness cases for the ported plasma geometry model (registry unit #24).

`plasma_angles_arcs`, `plasma_poloidal_perimeter`, `plasma_surface_area`,
`plasma_volume`, `plasma_cross_section` and `sauter_geometry` are diffed directly
against `PlasmaGeom`'s own `@staticmethod`s -- they take no `self.data` access at all,
so no adapter is needed and `reference = staticmethod(PlasmaGeom.<name>)` is exact.

`calculate_minor_radius`, `calculate_shape_ipdg89_x_point` and
`calculate_geometry_double_arc` have no PROCESS function of the same shape (`run()`'s
preamble, one `i_plasma_geometry` branch and the double-arc arm are each inline code,
not standalone methods), so each is diffed against a small adapter
(`_run_plasma_geom`) that builds a real `DataStructure` with `large_tokamak_eval.
IN.DAT`'s own switch values (`i_plasma_geometry=0`, `i_plasma_wall_gap=1`,
`i_plasma_current=4`, `i_plasma_shape=0`), calls the real, bound `PlasmaGeom.run()`,
and reads the relevant fields back -- the "close the `data` backdoor" technique used
throughout this harness, and the strongest oracle available here: it validates against
the actual stateful method, not a transcription of it.

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

from functional_process._harness import Tier1Contract, legacy_sample
from functional_process.models.physics.plasma_geometry import (
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


def _run_plasma_geom(rmajor, aspect, kappa, triang, f_vol_plasma=1.0):
    """Build a `DataStructure`, run the real `PlasmaGeom.run()` and return `data`.

    Switch values match `large_tokamak_eval.IN.DAT`: `i_plasma_geometry=0`
    (`IPDG89_X_POINT`), `i_plasma_wall_gap=1` (no build fields touched),
    `i_plasma_current=4`, `i_plasma_shape=0` (`PROCESS_ORIGINAL`, so the double-arc arm
    of the compound Sauter switch is taken).
    """
    data = DataStructure()
    data.physics.rmajor = rmajor
    data.physics.aspect = aspect
    data.physics.kappa = kappa
    data.physics.triang = triang
    data.physics.f_vol_plasma = f_vol_plasma
    data.physics.i_plasma_geometry = PlasmaGeometryModelType.IPDG89_X_POINT
    data.physics.i_plasma_wall_gap = 1
    data.physics.i_plasma_current = 4
    data.physics.i_plasma_shape = PlasmaShapeModelType.PROCESS_ORIGINAL

    pg = PlasmaGeom()
    pg.data = data
    pg.run()
    return data


def _run_plasma_geom_create_data_eu_demo(aspect, m_s_limit, triang):
    """Build a `DataStructure`, run the real `PlasmaGeom.run()` under
    `i_plasma_geometry = 10`, and return `data`.

    Switch values match `low_aspect_ratio_DEMO.IN.DAT` (the value-10 regression input):
    `i_plasma_geometry = 10` (`CREATE_DATA_EU_DEMO_X_POINT`, `:372`),
    `i_plasma_current = 4` (`:352`), `i_plasma_wall_gap` and `i_plasma_shape` unset
    (defaults `1` and `0`). `rmajor` is held at the file's own `8.6` (`:164`) -- it
    feeds only `rminor`/`eps`, not this branch's three outputs. The input file's
    `kappa = 1.848` initial value is deliberately *not* set: value 10 overwrites
    `kappa`, which is exactly the ownership this occupant claims.
    """
    data = DataStructure()
    data.physics.rmajor = 8.6
    data.physics.aspect = aspect
    data.physics.m_s_limit = m_s_limit
    data.physics.triang = triang
    data.physics.i_plasma_geometry = PlasmaGeometryModelType.CREATE_DATA_EU_DEMO_X_POINT
    data.physics.i_plasma_wall_gap = 1
    data.physics.i_plasma_current = 4
    data.physics.i_plasma_shape = PlasmaShapeModelType.PROCESS_ORIGINAL

    pg = PlasmaGeom()
    pg.data = data
    pg.run()
    return data


def _reference_calculate_shape_create_data_eu_demo_x_point(aspect, m_s_limit, triang):
    data = _run_plasma_geom_create_data_eu_demo(
        aspect=aspect, m_s_limit=m_s_limit, triang=triang
    )
    return data.physics.kappa95, data.physics.kappa, data.physics.triang95


def _reference_calculate_minor_radius(rmajor, aspect):
    data = _run_plasma_geom(rmajor=rmajor, aspect=aspect, kappa=1.7, triang=0.4)
    return data.physics.rminor, data.physics.eps


def _reference_calculate_shape_ipdg89_x_point(kappa, triang):
    data = _run_plasma_geom(rmajor=8.0, aspect=3.2, kappa=kappa, triang=triang)
    return data.physics.kappa95, data.physics.triang95


def _reference_calculate_geometry_double_arc(
    rmajor, rminor, kappa, triang, f_vol_plasma
):
    aspect = rmajor / rminor
    data = _run_plasma_geom(
        rmajor=rmajor,
        aspect=aspect,
        kappa=kappa,
        triang=triang,
        f_vol_plasma=f_vol_plasma,
    )
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

    samples = [
        legacy_sample(
            "plasma_angles_arcs-baseline_2018",
            a=2.8677741935483869,
            kappa=1.8480000000000001,
            triang=0.5,
        ),
    ]

    fuzz_bounds = {
        "a": (0.5, 5.0),
        "kappa": (1.6, 2.5),
        "triang": (0.0, 0.3),
    }


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

    samples = [
        legacy_sample(
            "plasma_poloidal_perimeter-baseline_2018",
            xi=10.510690667870968,
            thetai=0.52847258461252744,
            xo=5.4154130183225808,
            thetao=1.3636548755403939,
        ),
    ]

    fuzz_bounds = {
        "xi": (1.0, 20.0),
        "thetai": (0.1, 1.4),
        "xo": (1.0, 20.0),
        "thetao": (0.1, 1.4),
    }


class TestPlasmaSurfaceArea(Tier1Contract):
    """`plasma_surface_area` -> the same, unchanged.

    Sample is `tests/unit/models/physics/test_plasma_geom.py::test_plasma_surface_area`'s
    point, verbatim -- genuinely legacy.
    """

    audit_record = "models/physics/plasma_geometry.md"
    reference = staticmethod(PlasmaGeom.plasma_surface_area)
    ported = plasma_surface_area

    samples = [
        legacy_sample(
            "plasma_surface_area-baseline_2018",
            rmajor=8.8901000000000003,
            rminor=2.8677741935483869,
            xi=10.510690667870968,
            thetai=0.52847258461252744,
            xo=5.4154130183225808,
            thetao=1.3636548755403939,
        ),
    ]

    fuzz_bounds = {
        "rmajor": (2.0, 20.0),
        "rminor": (0.5, 5.0),
        "xi": (1.0, 20.0),
        "thetai": (0.1, 1.4),
        "xo": (1.0, 20.0),
        "thetao": (0.1, 1.4),
    }


class TestPlasmaVolume(Tier1Contract):
    """`plasma_volume` -> the same, unchanged.

    Sample is `tests/unit/models/physics/test_plasma_geom.py::test_plasma_volume`'s
    point, verbatim -- genuinely legacy.
    """

    audit_record = "models/physics/plasma_geometry.md"
    reference = staticmethod(PlasmaGeom.plasma_volume)
    ported = plasma_volume

    samples = [
        legacy_sample(
            "plasma_volume-baseline_2018",
            rmajor=9.2995201822511735,
            rminor=2.9998452200810237,
            xi=10.261919050584332,
            thetai=0.54748563700358688,
            xo=5.4205364969154601,
            thetao=1.4001019213417263,
        ),
    ]

    fuzz_bounds = {
        "rmajor": (2.0, 20.0),
        "rminor": (0.5, 5.0),
        "xi": (1.0, 20.0),
        "thetai": (0.1, 1.4),
        "xo": (1.0, 20.0),
        "thetao": (0.1, 1.4),
    }


class TestPlasmaCrossSection(Tier1Contract):
    """`plasma_cross_section` -> the same, unchanged.

    Sample is `tests/unit/models/physics/test_plasma_geom.py::
    test_plasma_cross_section`'s point, verbatim -- genuinely legacy.
    """

    audit_record = "models/physics/plasma_geometry.md"
    reference = staticmethod(PlasmaGeom.plasma_cross_section)
    ported = plasma_cross_section

    samples = [
        legacy_sample(
            "plasma_cross_section-baseline_2018",
            xi=10.261919050584332,
            thetai=0.54748563700358688,
            xo=5.4205364969154601,
            thetao=1.4001019213417263,
        ),
    ]

    fuzz_bounds = {
        "xi": (1.0, 20.0),
        "thetai": (0.1, 1.4),
        "xo": (1.0, 20.0),
        "thetao": (0.1, 1.4),
    }


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

    samples = [
        legacy_sample(
            "sauter_geometry-iter-scale",
            a=2.5,
            r0=8.0,
            kappa=1.85,
            triang=0.5,
            square=0.0,
        ),
    ]

    fuzz_bounds = {
        "a": (0.5, 5.0),
        "r0": (2.0, 20.0),
        "kappa": (1.6, 2.5),
        "triang": (0.0, 0.3),
        "square": (-0.2, 0.2),
    }


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

    samples = [
        legacy_sample(
            "calculate_geometry_sauter-iter-scale",
            rmajor=8.0,
            rminor=2.5,
            kappa=1.85,
            triang=0.5,
            plasma_square=0.0,
        ),
    ]

    fuzz_bounds = {
        "rmajor": (2.0, 20.0),
        "rminor": (0.5, 5.0),
        "kappa": (1.6, 2.5),
        "triang": (0.0, 0.3),
        "plasma_square": (-0.2, 0.2),
    }


class TestCalculateMinorRadius(Tier1Contract):
    """`calculate_minor_radius` -> `PlasmaGeom.run()`'s unconditional preamble.

    No standalone PROCESS function of this shape; diffed against `_run_plasma_geom`,
    which calls the real, bound `run()` and reads `.physics.rminor`/`.physics.eps` back.
    """

    audit_record = "models/physics/plasma_geometry.md"
    reference = staticmethod(_reference_calculate_minor_radius)
    ported = calculate_minor_radius

    samples = [
        legacy_sample(
            "calculate_minor_radius-large_tokamak_eval",
            rmajor=8.0,
            aspect=3.2,
        ),
    ]

    fuzz_bounds = {
        "rmajor": (2.0, 20.0),
        "aspect": (1.5, 4.0),
    }


class TestCalculateShapeIpdg89XPoint(Tier1Contract):
    """`calculate_shape_ipdg89_x_point` -> `i_plasma_geometry == IPDG89_X_POINT` (0).

    No standalone PROCESS function of this shape (it is 2 lines of `run()`'s dispatch);
    diffed against `_run_plasma_geom`, reading `.physics.kappa95`/`.physics.triang95`
    back after the real `run()` call.
    """

    audit_record = "models/physics/plasma_geometry.md"
    reference = staticmethod(_reference_calculate_shape_ipdg89_x_point)
    ported = calculate_shape_ipdg89_x_point

    samples = [
        legacy_sample(
            "calculate_shape_ipdg89_x_point-large_tokamak_eval",
            kappa=1.85,
            triang=0.5,
        ),
    ]

    fuzz_bounds = {
        "kappa": (1.6, 2.5),
        "triang": (0.0, 0.3),
    }


class TestCalculateShapeCreateDataEuDemoXPoint(Tier1Contract):
    """`calculate_shape_create_data_eu_demo_x_point` -> `i_plasma_geometry == 10`.

    No standalone PROCESS function of this shape (the branch is inline in `run()`'s
    dispatch, `plasma_geometry.py:362-397`); diffed against
    `_run_plasma_geom_create_data_eu_demo`, which calls the real, bound `run()` under
    `i_plasma_geometry = 10` and reads `.physics.kappa95`/`.kappa`/`.triang95` back.

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

    samples = [
        legacy_sample(
            "calculate_shape_create_data_eu_demo_x_point-low_aspect_ratio_DEMO",
            aspect=2.8,
            m_s_limit=0.2,
            triang=0.5,
        ),
        legacy_sample(
            "calculate_shape_create_data_eu_demo_x_point-corner-fudge-arm",
            aspect=2.6,
            m_s_limit=0.0,
            triang=0.4,
        ),
    ]

    fuzz_bounds = {
        "aspect": (2.6, 3.6),
        "m_s_limit": (0.0, 0.5),
        "triang": (0.0, 0.5),
    }


class TestCalculateGeometryDoubleArc(Tier1Contract):
    """`calculate_geometry_double_arc` -> the geometry-model arm's double-arc arm.

    No standalone PROCESS function of this shape; diffed against `_run_plasma_geom`
    (whose switch configuration always takes the double-arc arm), reading
    `.physics.len_plasma_poloidal`/`.vol_plasma`/`.a_plasma_poloidal`/`.a_plasma_surface`
    back after the real `run()` call. This is the node that owns three of the slot's
    five target outputs (`a_plasma_poloidal`, `a_plasma_surface`, `vol_plasma`).
    """

    audit_record = "models/physics/plasma_geometry.md"
    reference = staticmethod(_reference_calculate_geometry_double_arc)
    ported = calculate_geometry_double_arc

    samples = [
        legacy_sample(
            "calculate_geometry_double_arc-large_tokamak_eval",
            rmajor=8.0,
            rminor=2.5,
            kappa=1.85,
            triang=0.5,
            f_vol_plasma=1.0,
        ),
    ]

    fuzz_bounds = {
        "rmajor": (2.0, 20.0),
        "rminor": (0.5, 5.0),
        "kappa": (1.6, 2.5),
        "triang": (0.0, 0.3),
        "f_vol_plasma": (0.5, 1.5),
    }
