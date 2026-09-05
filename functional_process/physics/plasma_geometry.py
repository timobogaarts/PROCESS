"""Pure physics functions extracted from `models/physics/plasma_geometry.py`.

See that file for the declarations wiring these into the graph, and
`_audit/units/models/physics/plasma_geometry.md` for the audit record. No
graph-framework import belongs in this module -- see `_audit/formulas_split.md`.
"""

import jax.numpy as jnp

from functional_process.models.safe_math import safe_pow, safe_sqrt


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


def calculate_shape_create_data_eu_demo_x_point(aspect, m_s_limit, triang):
    """Elongations and 95%-surface triangularity from the CREATE-data EU-DEMO fit.

    Ports the `i_plasma_geometry == CREATE_DATA_EU_DEMO_X_POINT` (10) branch of
    `PlasmaGeom.run`, `process/models/physics/plasma_geometry.py:362-397`, unchanged:
    `kappa95` from a fit to CREATE data over `aspect` and the stability margin
    `m_s_limit` (PROCESS issues #1399/#1648, documented valid for aspect ratio
    2.6-3.6), a corner-fudge correction above `kappa95 == 1.77`, then
    `kappa = 1.12 * kappa95` and `triang95 = triang / 1.50`.

    Two of the audit record's JAX flags land here, both `workaround-known`:

    - **F1/D6**: the source's `if kappa95 > 1.77:` is a Python `if` on a freshly
      computed value -- here a `jnp.where`, which faithfully reproduces the value
      (both arms give `1.77` at the point, C0) *and* the derivative kink (measured
      one-sided derivatives 1.0000 from below, 0.7290 from above -- the branch is not
      C1 in PROCESS and is not made C1 here).
    - **F3**: the radicand of the fit's square root is sign-unconstrained -- nothing
      enforces the documented `2.6 < aspect < 3.6` validity range, and outside it the
      radicand can go negative. PROCESS's `np.sqrt` returns `nan` silently; this port
      does the same through `safe_sqrt` (identical values, finite derivative at an
      exactly-zero radicand). Likewise `kappa95 ** ratio` has a *traced* exponent, so
      it goes through `safe_pow` (F4).

    Returns
    -------
    tuple
        `(kappa95, kappa, triang95)` -- the branch's writes, in source order.
    """
    a = 3.68436807e0
    b = -0.27706527e0
    c = 0.87040251e0
    d = -18.83740952e0
    e = -0.27267618e0
    f = 20.5141261e0

    kappa95 = (
        -d
        - c * aspect
        - safe_sqrt(
            (c**2.0e0 - 4.0e0 * a * b) * aspect**2.0e0
            + (2.0e0 * d * c - 4.0e0 * a * e) * aspect
            + d**2.0e0
            - 4.0e0 * a * f
            + 4.0e0 * a * m_s_limit
        )
    ) / (2.0e0 * a)

    # `if kappa95 > 1.77:` in the source -- a traced-value branch (F1), so `jnp.where`
    # over both arms. The untaken arm is finite whenever `kappa95 > 0`, which holds
    # everywhere the fit itself is finite, so no tangent poisoning leaks through.
    ratio = 1.77e0 / kappa95
    corner_fudge = 0.3e0 * (kappa95 - 1.77e0) / ratio
    kappa95 = jnp.where(
        kappa95 > 1.77e0, safe_pow(kappa95, ratio) + corner_fudge, kappa95
    )

    kappa = 1.12e0 * kappa95
    triang95 = triang / 1.50e0
    return kappa95, kappa, triang95


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
