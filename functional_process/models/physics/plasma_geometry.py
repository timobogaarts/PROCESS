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

- 12 of 13 `i_plasma_geometry` values (1-12) -- none is live on any input this pass
  covers, and each needs different reads (`fkzohm`, `ind_plasma_internal_norm`,
  `m_s_limit`, ...). See the record's "switches touched" table for each value's reads
  and writes.
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

import jax.numpy as jnp
from cottax.interfaces.pytree_namespace_module import ExplicitFunction, From, OutputInto

from functional_process.paths import physics

# ---------------------------------------------------------------------------
# Functions 3-9 of the audit record's numbering: already pure `@staticmethod`s in
# `process/`, ported verbatim (`np.` -> `jnp.`). Zero `self.data` access in the source.
# ---------------------------------------------------------------------------


def plasma_angles_arcs(a, kappa, triang):
    """Parameters of the two arcs describing the plasma cross-section.

    Ports `PlasmaGeom.plasma_angles_arcs`,
    `process/models/physics/plasma_geometry.py:711-759`, unchanged.

    **Reproduces D1 faithfully, does not fix it.** The audit record's "suspected
    defects" **D1** (confirmed by measurement): for `kappa < 1 + triang`, `denomo`
    goes negative and `arctan` returns the wrong branch, silently flipping the sign of
    every downstream quantity (perimeter, cross-section, volume, surface). Exactly at
    `kappa == 1 + triang` or `triang == +-1.0` the source divides by zero. PROCESS
    itself has no guard here (no exception, no warning) and this port has none either --
    the precondition `kappa > 1 + triang` is the caller's to hold, same as in PROCESS.
    Samples in this unit's test file are chosen to respect it (see `_audit/units/
    models/physics/plasma_geometry.md`'s open question 3).

    Returns
    -------
    tuple
        `(xi, thetai, xo, thetao)` -- inboard/outboard arc radius and half-angle.
    """
    t = 1.0 - triang
    denomi = (kappa**2 - t**2) / (2.0 * t)
    thetai = jnp.arctan(kappa / denomi)
    xi = a * (denomi + 1.0 - triang)

    n = 1.0 + triang
    denomo = (kappa**2 - n**2) / (2.0 * n)
    thetao = jnp.arctan(kappa / denomo)
    xo = a * (denomo + 1.0 + triang)

    return xi, thetai, xo, thetao


def plasma_poloidal_perimeter(xi, thetai, xo, thetao):
    """Plasma poloidal perimeter (m). Ports `PlasmaGeom.plasma_poloidal_perimeter`,
    `process/models/physics/plasma_geometry.py:761-783`, unchanged.
    """
    return 2.0 * (xo * thetao + xi * thetai)


def plasma_surface_area(rmajor, rminor, xi, thetai, xo, thetao):
    """Inboard and outboard plasma surface area (m^2). Ports `PlasmaGeom.
    plasma_surface_area`, `process/models/physics/plasma_geometry.py:785-830`,
    unchanged.

    Returns
    -------
    tuple
        `(xsi, xso)`.
    """
    fourpi = 4.0 * jnp.pi

    rc = rmajor - rminor + xi
    xsi = fourpi * xi * (rc * thetai - xi * jnp.sin(thetai))

    rc = rmajor + rminor - xo
    xso = fourpi * xo * (rc * thetao + xo * jnp.sin(thetao))

    return xsi, xso


def plasma_volume(rmajor, rminor, xi, thetai, xo, thetao):
    """Plasma volume (m^3). Ports `PlasmaGeom.plasma_volume`,
    `process/models/physics/plasma_geometry.py:832-896`, unchanged.
    """
    third = 1.0 / 3.0

    rc = rmajor - rminor + xi
    vin = (
        2.0
        * jnp.pi
        * xi
        * (
            rc**2 * jnp.sin(thetai)
            - rc * xi * thetai
            - 0.5 * rc * xi * jnp.sin(2.0 * thetai)
            + xi * xi * jnp.sin(thetai)
            - third * xi * xi * (jnp.sin(thetai)) ** 3
        )
    )

    rc = rmajor + rminor - xo
    vout = (
        2.0
        * jnp.pi
        * xo
        * (
            rc**2 * jnp.sin(thetao)
            + rc * xo * thetao
            + 0.5 * rc * xo * jnp.sin(2.0 * thetao)
            + xo * xo * jnp.sin(thetao)
            - third * xo * xo * (jnp.sin(thetao)) ** 3
        )
    )

    return vout - vin


def plasma_cross_section(xi, thetai, xo, thetao):
    """Plasma cross-sectional area (m^2). Ports `PlasmaGeom.plasma_cross_section`,
    `process/models/physics/plasma_geometry.py:898-931`, unchanged.
    """
    return xo**2 * (thetao - jnp.cos(thetao) * jnp.sin(thetao)) + xi**2 * (
        thetai - jnp.cos(thetai) * jnp.sin(thetai)
    )


def sauter_geometry(a, r0, kappa, triang, square):
    """Sauter-model plasma geometry. Ports `PlasmaGeom.sauter_geometry`,
    `process/models/physics/plasma_geometry.py:933-1001`, unchanged.

    Ported for completeness (the audit record recommends porting functions 3-9
    verbatim regardless), but **not wired to an occupant class in this pass** -- the
    compound switch that selects it (`i_plasma_current == 8 or i_plasma_shape ==
    SAUTER`) is not live on `large_tokamak_eval.IN.DAT` and the audit record notes it
    "has no regression oracle at all" among tracked inputs.

    Returns
    -------
    tuple
        `(len_plasma_poloidal, a_plasma_surface, a_plasma_poloidal, vol_plasma)`.
    """
    w07 = square + 1.0
    eps = a / r0

    len_plasma_poloidal = (
        2.0
        * jnp.pi
        * a
        * (1.0 + 0.55 * (kappa - 1.0))
        * (1.0 + 0.08 * triang**2)
        * (1.0 + 0.2 * (w07 - 1.0))
    )

    a_plasma_surface = (
        2.0 * jnp.pi * r0 * (1.0 - 0.32 * triang * eps) * len_plasma_poloidal
    )

    a_plasma_poloidal = jnp.pi * a**2 * kappa * (1.0 + 0.52 * (w07 - 1.0))

    vol_plasma = 2.0 * jnp.pi * r0 * (1.0 - 0.25 * triang * eps) * a_plasma_poloidal

    return len_plasma_poloidal, a_plasma_surface, a_plasma_poloidal, vol_plasma


# ---------------------------------------------------------------------------
# New: extracted from `run()`'s unconditional preamble
# (`process/models/physics/plasma_geometry.py:224-227`). No switch involved.
# ---------------------------------------------------------------------------


def calculate_minor_radius(rmajor, aspect):
    """Plasma minor radius and inverse aspect ratio. Extracted from `PlasmaGeom.run`'s
    unconditional preamble, `process/models/physics/plasma_geometry.py:224-227`.

    Unconditional in `process/` -- no switch decides this, so there is one occupant and
    no family. This file is the sole tokamak producer of both outputs (audit record's
    data footprint table).

    Returns
    -------
    tuple
        `(rminor, eps)`.
    """
    rminor = rmajor / aspect
    eps = 1.0 / aspect
    return rminor, eps


# ---------------------------------------------------------------------------
# New: the `i_plasma_geometry == IPDG89_X_POINT` (0) branch,
# `process/models/physics/plasma_geometry.py:231-241`. The only `i_plasma_geometry`
# value this pass ports -- see the module docstring for the other twelve.
# ---------------------------------------------------------------------------


def calculate_shape_ipdg89_x_point(kappa, triang):
    """95%-surface elongation and triangularity, IPDG89 fit. Ports the
    `i_plasma_geometry == IPDG89_X_POINT` (0) branch of `PlasmaGeom.run`,
    `process/models/physics/plasma_geometry.py:231-241`, unchanged.

    `kappa`/`triang` are read, not written, under this branch (`PlasmaGeometryModelType.
    IPDG89_X_POINT.kappa_model == triang_model == USER_INPUT` -- the audit record's "the
    enum is a machine-readable ownership table"): under `i_plasma_geometry == 0` they are
    plain boundary inputs, not produced by any node in this file.

    Returns
    -------
    tuple
        `(kappa95, triang95)`.
    """
    kappa95 = kappa / 1.12
    triang95 = triang / 1.50
    return kappa95, triang95


# ---------------------------------------------------------------------------
# New: the geometry-model arm, `process/models/physics/plasma_geometry.py:445-509`.
# The compound switch `i_plasma_current == 8 or i_plasma_shape == SAUTER` collapses to
# one boolean at graph-assembly time (audit record's open question 2) -- this pass
# ports only the `False` (double-arc) arm, the one `large_tokamak_eval.IN.DAT` takes.
# ---------------------------------------------------------------------------


def calculate_geometry_double_arc(rmajor, rminor, kappa, triang, f_vol_plasma):
    """Poloidal perimeter, volume, cross-section and surface area, double-arc model.

    Ports the `else` (non-Sauter) arm of `PlasmaGeom.run`'s geometry-model `if`,
    `process/models/physics/plasma_geometry.py:445-461,484-509` -- the arcs/surface-area
    preamble that is shared with the Sauter arm plus the double-arc-specific perimeter,
    volume and cross-section. Composes the functions above; introduces no new
    arithmetic of its own.

    `f_vol_plasma` is a plain user-settable volume multiplier (default `1.0`, never
    assigned by any model in `process/` -- the audit record's **D2**), so it is a
    boundary read, not a switch.

    Returns
    -------
    tuple
        `(len_plasma_poloidal, vol_plasma, a_plasma_poloidal, a_plasma_surface)`.
    """
    xi, thetai, xo, thetao = plasma_angles_arcs(rminor, kappa, triang)
    xsi, xso = plasma_surface_area(rmajor, rminor, xi, thetai, xo, thetao)

    len_plasma_poloidal = plasma_poloidal_perimeter(xi, thetai, xo, thetao)
    vol_plasma = f_vol_plasma * plasma_volume(rmajor, rminor, xi, thetai, xo, thetao)
    a_plasma_poloidal = plasma_cross_section(xi, thetai, xo, thetao)
    a_plasma_surface = xsi + xso

    return len_plasma_poloidal, vol_plasma, a_plasma_poloidal, a_plasma_surface


def calculate_geometry_sauter(rmajor, rminor, kappa, triang, plasma_square):
    """Poloidal perimeter, volume, cross-section and surface area, Sauter model.

    Ports the `if` (Sauter) arm of `PlasmaGeom.run`'s geometry-model `if`,
    `process/models/physics/plasma_geometry.py:467-482`, reordered to the same
    `(len_plasma_poloidal, vol_plasma, a_plasma_poloidal, a_plasma_surface)` tuple shape
    as `calculate_geometry_double_arc` for symmetry (`sauter_geometry` itself returns
    `a_plasma_surface` before `a_plasma_poloidal`; unchanged there, only reordered at
    this composition).

    Ported for completeness (see `sauter_geometry`'s docstring); **not wired to an
    occupant class in this pass** -- not live on `large_tokamak_eval.IN.DAT`.

    Returns
    -------
    tuple
        `(len_plasma_poloidal, vol_plasma, a_plasma_poloidal, a_plasma_surface)`.
    """
    len_plasma_poloidal, a_plasma_surface, a_plasma_poloidal, vol_plasma = (
        sauter_geometry(rminor, rmajor, kappa, triang, plasma_square)
    )
    return len_plasma_poloidal, vol_plasma, a_plasma_poloidal, a_plasma_surface


# ---------------------------------------------------------------------------
# cottax nodes
# ---------------------------------------------------------------------------


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
