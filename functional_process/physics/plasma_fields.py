"""Pure physics functions extracted from `models/physics/plasma_fields.py`.

See that file for the declarations wiring these into the graph, and
`_audit/units/models/physics/plasma_fields.md` for the audit record. No
graph-framework import belongs in this module -- see `_audit/formulas_split.md`.
"""

import jax.numpy as jnp


def calculate_plasma_inboard_toroidal_field(b_plasma_toroidal_on_axis, rmajor, rminor):
    """Toroidal field at the plasma inboard midplane (Bᴛ(R₀-a)).

    Ports `PlasmaFields.calculate_plasma_inboard_toroidal_field`,
    `process/models/physics/plasma_fields.py:95-118`, unchanged -- a single division,
    no fractional power, no `safe_*` site needed. Singular at `rmajor == rminor`
    (aspect ratio 1, outside every tracked regression input and outside
    `ITERATION_VARIABLES`' own bounds for `aspect`/`rmajor`/`rminor`), matching
    PROCESS's own unguarded division exactly.
    """
    return rmajor * b_plasma_toroidal_on_axis / (rmajor - rminor)


def calculate_plasma_outboard_toroidal_field(b_plasma_toroidal_on_axis, rmajor, rminor):
    """Toroidal field at the plasma outboard midplane (Bᴛ(R₀+a)).

    Ports `PlasmaFields.calculate_plasma_outboard_toroidal_field`,
    `process/models/physics/plasma_fields.py:120-143`, unchanged. No singularity on the
    physical domain (`rmajor + rminor` is a sum of two positive lengths).
    """
    return rmajor * b_plasma_toroidal_on_axis / (rmajor + rminor)


def calculate_toroidal_field_profile(
    b_plasma_toroidal_on_axis, rmajor, rminor, n_plasma_profile_elements
):
    """Toroidal field profile across the plasma midplane (1/R dependence).

    Ports `PlasmaFields.calculate_toroidal_field_profile`,
    `process/models/physics/plasma_fields.py:145-177`, `np.` -> `jnp.` only.
    `n_plasma_profile_elements` sizes the returned array (`2 * n_plasma_profile_elements`
    points, default 201 -> 402) and must stay a concrete Python `int` under
    `jax.jit`/`jacfwd` -- it is a profile-resolution *count*, not a switch, but it is
    exactly the "dynamic shape" hazard `_audit/traceability_policy.md` flags. Whoever
    wires this into a node must declare it `static_argnames`, not a traced `VarPath`.

    **Ported, not wired to an occupant.** `.physics.b_plasma_toroidal_profile` has no
    reader anywhere in the currently-assembled tokamak graph. Its only consumers in
    `process/` are deep inside `Physics.run()` itself
    (`physics.py:3860-3872`'s `beta_thermal_toroidal_profile` and the six
    Larmor-frequency profile computations at `physics.py:5284-5330`), none of which is
    ported -- `physics.py` is the ~7000-line file `CLAUDE.md`'s difficulty list names
    explicitly. Ported now because it is part of this file's closure and costs nothing
    extra to test; the node is one class away whenever a real consumer exists, the same
    deferral this file already used for the inboard/outboard totals before this pass.
    """
    rho = jnp.linspace(rmajor - rminor, rmajor + rminor, 2 * n_plasma_profile_elements)
    rho = jnp.where(rho == 0, 1e-10, rho)
    return rmajor * b_plasma_toroidal_on_axis / rho
