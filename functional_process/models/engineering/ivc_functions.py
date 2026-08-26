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

**Only the elliptical pair is ported.** `dshellarea`/`dshellvol` (the D-shaped
counterparts) are not needed by either occupant class currently written against this
file -- both `first_wall.py` and `vacuum.py`'s wave-1 ports only wire up the elliptical
arm, since that is the one the reference input reaches (`itart == 1 or
i_fw_blkt_vv_shape == D_SHAPED` is `False` there). Add them here, not privately, the day
a D-shaped occupant is written.

Both functions are already pure in `process/` -- no `self.data` access, `@staticmethod`-
free module-level `def`s -- so this is a mechanical `np.` -> `jnp.` port, no signature
change, no switch to resolve, no `self.data` backdoor to close. No legacy sample exists
in `tests/unit/` for either function (`grep -rl eshellarea|eshellvol tests/unit` is
empty); the fw/vacuum-vessel units' own legacy samples (`test_fw.py`'s D-DEMO-shaped
values inherited via `calculate_elliptical_first_wall_areas` are not present either --
`tests/unit/models/test_fw.py` only exercises `fw_temp`, the CoolProp arm, out of
scope) exercise these formulas indirectly instead:
`tests/unit/models/test_vacuum.py::test_elliptical_vessel_volumes` is this file's only
free oracle, reused as this unit's legacy sample too.
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
