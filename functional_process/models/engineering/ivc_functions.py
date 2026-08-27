"""Pure-functional port of `process/models/engineering/ivc_functions.py` (partial).

Shared home for the toroidal-shell area/volume formulas used by more than one model --
per the wave-1 tokamak dispatch's own convention: a helper needed by two or more units
is ported once, here, rather than duplicated privately into each unit's own module. Not
itself one of `_audit/unit_registry.md`'s numbered units; it exists because
`.tokamak.first_wall` (`models/fw.py::FirstWall.calculate_elliptical_first_wall_areas`)
and `.tokamak.vacuum_vessel` (`models/vacuum.py::VacuumVessel.
calculate_elliptical_vessel_volumes`) both call into this file's elliptical shell
formulas on the live `large_tokamak_eval.IN.DAT` path (`itart = 0`,
`.fwbs.i_fw_blkt_vv_shape` defaults to `2`, `ELLIPTICAL_SHAPED` --
`process/models/build.py:26`).

**2026-08-27 (the D-shaped wave): all four are ported.** Until this date only the
elliptical pair was, and the docstring here said *"add them here, not privately, the day
a D-shaped occupant is written"*. That day is this one: `spherical_tokamak_eval.IN.DAT`
and `st_regression.IN.DAT` both set `i_fw_blkt_vv_shape = 1` (`D_SHAPED`) **and**
`itart = 1`, so the disjunction `itart == 1 or i_fw_blkt_vv_shape == D_SHAPED`
(`process/models/blankets/blanket_library.py:90-93`, `fw.py:58-86`, `shield.py:47-50`,
`vacuum.py:757-760`) is doubly true and five slots at once needed the D-shaped arm.
`dshellarea` and `dshellvol` are added below beside their elliptical siblings, and
`models/shield.py`'s two private copies (`_eshellvol`/`_dshellvol`, filed there in wave 1
with an explicit note that the consolidation pass should lift them) now import from here
instead.

`models/blankets/blanket_library.py` still keeps `_eshellarea`/`_eshellvol` private, for
the reason its own docstrings give -- left alone deliberately, so that this wave changes
one file's filing and not two.

All four functions are already pure in `process/` -- no `self.data` access,
`@staticmethod`-free module-level `def`s -- so this is a mechanical `np.` -> `jnp.` port,
no signature change, no switch to resolve, no `self.data` backdoor to close. No legacy
sample exists in `tests/unit/` for any of them (`grep -rl 'eshellarea|eshellvol|
dshellarea|dshellvol' tests/unit` is empty); the fw/vacuum-vessel/shield/blanket
units' own legacy samples exercise these formulas indirectly instead
(`tests/unit/models/test_vacuum.py::test_elliptical_vessel_volumes` is this file's only
free oracle, reused as this unit's legacy sample too).

**The D-shaped pair is not the elliptical pair with a parameter changed.** A D-shaped
shell's inboard section is a *cylinder* -- `ain` and `vin` are closed-form cylinder
formulas, not the difference of two ellipse revolutions -- and its outboard ellipse is
centred on the outer edge of that cylinder rather than at a shared `rshell`. So the two
pairs share a name and a return shape and nothing else, and neither can be expressed as
the other under a substitution. That is why they are four functions and not two with a
switch.
"""

import jax.numpy as jnp


def eshellarea(rshell, rmini, rmino, zminor):
    """Inboard, outboard and total surface areas of a toroidal shell made of two
    elliptical sections. Ports `process/models/engineering/ivc_functions.py:99-130`,
    unchanged.

    Parameters
    ----------
    rshell :
        Major radius of the centre of both ellipses (m).
    rmini :
        Horizontal distance from `rshell` to the inboard elliptical shell (m).
    rmino :
        Horizontal distance from `rshell` to the outboard elliptical shell (m).
    zminor :
        Vertical internal half-height of the shell (m).

    Returns
    -------
    tuple
        `(ain, aout, atot)` -- inboard, outboard and total surface area (m^2).
    """
    elong_in = zminor / rmini
    ain = 2.0 * jnp.pi * elong_in * (jnp.pi * rshell * rmini - 2.0 * rmini * rmini)

    elong_out = zminor / rmino
    aout = 2.0 * jnp.pi * elong_out * (jnp.pi * rshell * rmino + 2.0 * rmino * rmino)

    return ain, aout, ain + aout


def dshellarea(rmajor, rminor, zminor):
    """Inboard, outboard and total surface areas of a D-shaped toroidal shell. Ports
    `process/models/engineering/ivc_functions.py:133-167`, unchanged.

    The inboard section is a cylinder of radius `rmajor` and full height `2 * zminor`;
    the outboard section is a semi-ellipse centred on that cylinder's radius. Compare
    `eshellarea`, whose *both* sections are ellipse revolutions about a shared `rshell`
    -- the inboard halves of the two formulas are not related by any substitution.

    Parameters
    ----------
    rmajor :
        Major radius of the inboard straight (cylindrical) section (m).
    rminor :
        Horizontal internal width of the shell (m) -- the outboard semi-ellipse's
        horizontal semi-axis.
    zminor :
        Vertical half-height of the shell (m).

    Returns
    -------
    tuple
        `(ain, aout, atot)` -- inboard, outboard and total surface area (m^2).
    """
    # Inboard cylindrical shell: circumference * height.
    ain = 4.0 * zminor * jnp.pi * rmajor

    # Outboard elliptical section -- the same expression as `eshellarea`'s outboard
    # half, with `rmajor` in place of `rshell`.
    elong = zminor / rminor
    aout = 2.0 * jnp.pi * elong * (jnp.pi * rmajor * rminor + 2.0 * rminor * rminor)

    return ain, aout, ain + aout


def eshellvol(rshell, rmini, rmino, zminor, drin, drout, dz):
    """Inboard, outboard and total volumes of a toroidal shell made of two elliptical
    sections. Ports `process/models/engineering/ivc_functions.py:170-247`, unchanged.

    Each section's volume is the difference of the volumes of revolution enclosed by
    its inner and outer bounding semi-ellipses.

    Parameters
    ----------
    rshell :
        Major radius of the centre of both ellipses (m).
    rmini :
        Horizontal distance from `rshell` to the outer edge of the inboard elliptical
        shell (m).
    rmino :
        Horizontal distance from `rshell` to the inner edge of the outboard elliptical
        shell (m).
    zminor :
        Vertical internal half-height of the shell (m).
    drin :
        Horizontal thickness of the inboard shell at the midplane (m).
    drout :
        Horizontal thickness of the outboard shell at the midplane (m).
    dz :
        Vertical thickness of the shell at top/bottom (m).

    Returns
    -------
    tuple
        `(vin, vout, vtot)` -- inboard, outboard and total volume (m^3).
    """
    # Inboard section.
    a = rmini
    b = zminor
    elong = b / a
    v1 = 2.0 * jnp.pi * elong * (0.5 * jnp.pi * rshell * a**2 - (2.0 / 3.0) * a**3)

    a = rmini + drin
    b = zminor + dz
    elong = b / a
    v2 = 2.0 * jnp.pi * elong * (0.5 * jnp.pi * rshell * a**2 - (2.0 / 3.0) * a**3)

    vin = v2 - v1

    # Outboard section.
    a = rmino
    b = zminor
    elong = b / a
    v1 = 2.0 * jnp.pi * elong * (0.5 * jnp.pi * rshell * a**2 + (2.0 / 3.0) * a**3)

    a = rmino + drout
    b = zminor + dz
    elong = b / a
    v2 = 2.0 * jnp.pi * elong * (0.5 * jnp.pi * rshell * a**2 + (2.0 / 3.0) * a**3)

    vout = v2 - v1

    return vin, vout, vin + vout


def dshellvol(rmajor, rminor, zminor, drin, drout, dz):
    """Inboard, outboard and total volumes of a D-shaped toroidal shell. Ports
    `process/models/engineering/ivc_functions.py:249-306`, unchanged.

    The inboard section is a cylinder of uniform thickness `drin` and full height
    `2 * (zminor + dz)`, so its volume is closed-form rather than a difference of two
    revolutions. The outboard section *is* such a difference, between two semi-ellipses
    centred on the outer edge of the inboard cylinder -- the same two terms as
    `eshellvol`'s outboard half, with `rmajor` for `rshell`.

    Parameters
    ----------
    rmajor :
        Major radius to the outer point of the inboard straight section (m).
    rminor :
        Horizontal internal width of the shell (m).
    zminor :
        Vertical internal half-height of the shell (m).
    drin :
        Horizontal thickness of the inboard shell at the midplane (m).
    drout :
        Horizontal thickness of the outboard shell at the midplane (m).
    dz :
        Vertical thickness of the shell at top/bottom (m).

    Returns
    -------
    tuple
        `(vin, vout, vtot)` -- inboard, outboard and total volume (m^3).
    """
    # Inboard cylindrical shell -- an annulus of thickness `drin`, extruded.
    vin = 2.0 * (zminor + dz) * jnp.pi * (rmajor**2 - (rmajor - drin) ** 2)

    # Outboard elliptical section.
    a = rminor
    b = zminor
    elong = b / a
    v1 = 2.0 * jnp.pi * elong * (0.5 * jnp.pi * rmajor * a**2 + (2.0 / 3.0) * a**3)

    a = rminor + drout
    b = zminor + dz
    elong = b / a
    v2 = 2.0 * jnp.pi * elong * (0.5 * jnp.pi * rmajor * a**2 + (2.0 / 3.0) * a**3)

    vout = v2 - v1

    return vin, vout, vin + vout
