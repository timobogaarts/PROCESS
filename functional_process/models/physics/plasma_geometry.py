"""Pure-functional port of `process/models/physics/plasma_geometry.py`.

Registry unit #24. Audit record:
`functional_process/_audit/units/models/physics/plasma_geometry.md`. Read it first —
especially "the extraction seam" (functions 3-9 already need no signature change: no
`self.data` access at all) and "suspected defects in PROCESS" **D1**, which this port
reproduces faithfully rather than fixes (see `plasma_angles_arcs`'s docstring).

**Scope of this pass.** `_audit/tokamak_boundary.md`'s `.tokamak.plasma_geom` slot lists
five target outputs -- `.physics.a_plasma_poloidal`, `.physics.a_plasma_surface`,
`.physics.eps`, `.physics.rminor`, `.physics.vol_plasma` -- and `vol_plasma` alone has
eleven readers, the most of any row in that table. This file ports the minimal closure
that produces them on `tests/regression/input_files/large_tokamak_eval.IN.DAT`'s own
configuration (`i_plasma_geometry=0`, `i_plasma_wall_gap=1` (default),
`i_plasma_current=4`, `i_plasma_shape` unset (default `0`)), plus the one sibling
computation (`kappa95`/`triang95`) that shares the same `i_plasma_geometry` branch and is
therefore "the same unit" by the audit record's own accounting, even though it is not one
of the five listed outputs.

**Not ported, and why (all per the audit record's own tables):**

- 11 of 13 `i_plasma_geometry` values (1-9, 11, 12) -- none is live on any tracked
  input, and each needs different reads (`fkzohm`, `ind_plasma_internal_norm`, ...).
  See the record's "switches touched" table for each value's reads and writes. Value
  10 (`CREATE_DATA_EU_DEMO_X_POINT`) was added 2026-08-27 because it *is* live -- on
  `tests/regression/input_files/low_aspect_ratio_DEMO.IN.DAT` (`:372`) -- see
  `calculate_shape_create_data_eu_demo_x_point` below.
- The Sauter arm of the compound
  `i_plasma_current == 8 or i_plasma_shape == SAUTER` switch (`plasma_geometry.py:
  467-469`) -- not live here, and the record notes it "has no regression oracle at all."
  `sauter_geometry` itself is still ported below (it needs no `self.data` and the audit
  record recommends porting functions 3-9 verbatim regardless of which arm is wired),
  just not wired to an occupant class yet.
- `i_plasma_wall_gap == 0` (writes `.build.dr_fw_plasma_gap_{inboard,outboard}`) -- not
  live (`large_tokamak_eval` leaves the switch at its default, `1`, which reads and
  writes nothing in this file: `.build.dr_fw_plasma_gap_*` are then plain boundary
  inputs read directly by `build`). Nothing to port for the live value; the `==0` arm is
  reported UNPORTED.
- `.physics.a_plasma_surface_outboard` (`plasma_geometry.py:461`, written unconditionally
  from the double-arc arcs even under Sauter -- **D10**) -- not one of the five target
  outputs, consumed by `models/blankets/dcll.py`, out of this unit's scope.
- `calculate_iter_physics_basis_elongation` -- **already ported**, by unit #10
  (`functional_process/models/physics/confinement_time.py`), whose own module docstring
  records that `PlasmaGeom.calculate_iter_physics_basis_elongation` is called from
  *its* body, one line, no further dependencies. Not re-ported here to avoid a duplicate
  definition; see that file if this one's `kappa_ipb` is ever needed.
- `PlasmaGeom.output()` and the four dead legacy module-level functions (`surfa`,
  `perim`, `fvol`, `xsect0`) -- reporting-only and dead-in-`process/` respectively; the
  audit record's **D11** recommends the port simply not carry the legacy four.

Every switch touched by this file (`i_plasma_geometry`, `i_plasma_wall_gap`,
`i_plasma_current`, `i_plasma_shape`) is consumed only by *which occupant class exists*,
never inside a function body -- `_audit/naming_convention.md` § "switches are not
ports". None of the functions below takes an `i_*` argument.
"""

from cottax.interfaces.pytree_namespace_module import ExplicitFunction, From, OutputInto

from functional_process.paths import physics
from functional_process.physics.plasma_geometry import (
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

__all__ = [
    "calculate_geometry_sauter",
    "plasma_angles_arcs",
    "plasma_cross_section",
    "plasma_poloidal_perimeter",
    "plasma_surface_area",
    "plasma_volume",
    "sauter_geometry",
]


class PlasmaMinorRadius(ExplicitFunction):
    """cottax node: `calculate_minor_radius`, ports declared.

    Unconditional -- no switch, no family, matching `calculate_minor_radius`'s own
    docstring.
    """

    rminor = OutputInto(physics)
    eps = OutputInto(physics)

    def __call__(self, rmajor=From(physics), aspect=From(physics)):
        return calculate_minor_radius(rmajor, aspect)


class PlasmaShapeKappa95Triang95(ExplicitFunction):
    """The family that owns `.physics.kappa95`/`.physics.triang95` under
    `i_plasma_geometry`: one occupant per value, per `_audit/traceability_policy.md`'s
    split default -- **this pass ports only `IPDG89_X_POINT` (0)**, the value
    `large_tokamak_eval.IN.DAT` uses. The other twelve values are UNPORTED; see the
    module docstring and the audit record's "switches touched" table for each one's
    reads.
    """


class Ipdg89XPointPlasmaShape(PlasmaShapeKappa95Triang95):
    """`i_plasma_geometry == IPDG89_X_POINT` (0). Reads `kappa`/`triang` as plain
    boundary inputs -- under this branch neither is written by this file (the enum's
    `kappa_model`/`triang_model` are both `USER_INPUT`; see `calculate_shape_ipdg89_
    x_point`'s docstring).
    """

    kappa95 = OutputInto(physics)
    triang95 = OutputInto(physics)

    def __call__(self, kappa=From(physics), triang=From(physics)):
        return calculate_shape_ipdg89_x_point(kappa, triang)


class CreateDataEuDemoXPointPlasmaShape(PlasmaShapeKappa95Triang95):
    """`i_plasma_geometry == CREATE_DATA_EU_DEMO_X_POINT` (10). Live on
    `tests/regression/input_files/low_aspect_ratio_DEMO.IN.DAT` (`:372`).

    Unlike the IPDG89 sibling this occupant **owns `kappa` too** -- under value 10 the
    enum's ownership row is `kappa_model == IPDG89` (computed from `kappa95`),
    `triang_model == USER_INPUT` (read), and the audit record's dispatch table for
    value 10 lists reads `{aspect, m_s_limit, triang}` and writes
    `{kappa95, kappa, triang95}`. The family's name records the two fields *every*
    occupant owns; this one adds a third, which the record's
    "conditional-ownership-by-run-config" finding says is the expected shape here.
    """

    kappa95 = OutputInto(physics)
    kappa = OutputInto(physics)
    triang95 = OutputInto(physics)

    def __call__(
        self,
        aspect=From(physics),
        m_s_limit=From(physics),
        triang=From(physics),
    ):
        return calculate_shape_create_data_eu_demo_x_point(aspect, m_s_limit, triang)


class PlasmaGeometryArm(ExplicitFunction):
    """The family that owns `.physics.len_plasma_poloidal`, `.vol_plasma`,
    `.a_plasma_poloidal`, `.a_plasma_surface`: one occupant per arm of the compound
    switch `i_plasma_current == 8 or i_plasma_shape == SAUTER`
    (`process/models/physics/plasma_geometry.py:467-469`).

    **This pass ports only the `False` (double-arc) arm** -- `large_tokamak_eval.
    IN.DAT` sets `i_plasma_current = 4` and leaves `i_plasma_shape` at its default (`0`,
    `PROCESS_ORIGINAL`), so neither half of the disjunction is true. The Sauter arm
    (`True`) is UNPORTED: not live on any tracked regression input and, per the audit
    record, without a regression oracle at all.

    Per the audit record's open question 2, the two switches collapse to one boolean at
    graph-assembly time -- `i_plasma_current`'s own topology split
    (`plasma_current.py`, another unit's scope) and this file's split on the same
    disjunction need to be resolved together by whichever pass wires both; flagged in
    the final report, not decided here.
    """


class DoubleArcPlasmaGeometry(PlasmaGeometryArm):
    """`i_plasma_current != 8 and i_plasma_shape != SAUTER` -- the arm
    `large_tokamak_eval.IN.DAT` takes.
    """

    len_plasma_poloidal = OutputInto(physics)
    vol_plasma = OutputInto(physics)
    a_plasma_poloidal = OutputInto(physics)
    a_plasma_surface = OutputInto(physics)

    def __call__(
        self,
        rmajor=From(physics),
        rminor=From(physics),
        kappa=From(physics),
        triang=From(physics),
        f_vol_plasma=From(physics),
    ):
        return calculate_geometry_double_arc(rmajor, rminor, kappa, triang, f_vol_plasma)
