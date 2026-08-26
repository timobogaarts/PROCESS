"""The tokamak's magnetic-field closure: toroidal-field components plus every total
field `Physics.run` derives from them.

**Extends, does not replace, the consolidation-pass module.** The original module
(one node, `TotalMagneticField` owning `.physics.b_plasma_total`) is unchanged below;
this pass adds the rest of `process/models/physics/plasma_fields.py::PlasmaFields`'
tokamak-relevant closure -- the inboard/outboard toroidal-field components and the two
sibling total-field nodes `Physics.run` derives from them
(`process/models/physics/plasma_fields.py:95-177`, called from
`process/models/physics/physics.py:372-421`). `calculate_surface_averaged_poloidal_
field`'s Peng arm (`i_plasma_current == 2`, `plasma_fields.py:27-93`) remains UNPORTED,
exactly as `physics.py`'s `SurfaceAveragedPoloidalField` family already declares --
that switch belongs to `physics.py`, not here, and is untouched by this pass.

**The formula is imported, not rewritten.** `process/models/physics/plasma_fields.py::
PlasmaFields.calculate_total_magnetic_field` is `sqrt(b_toroidal**2 + b_poloidal**2)`,
and `models/stellarator/plasma_physics.py::calculate_total_field` is the same
expression, already ported and already harness-tested
(`tests/functional_process/models/stellarator/test_plasma_physics.py`). Every total-field
node below -- on-axis, inboard, outboard -- calls that one function; re-deriving it a
second or third time would be exactly the transcription defect this port keeps finding.

**A genuine PROCESS ordering bug, not reproduced.** `physics.py:378-392` computes
`b_plasma_outboard_total` and `b_plasma_inboard_total` *before* `physics.py:394-409`
computes the `b_plasma_outboard_toroidal`/`b_plasma_inboard_toroidal` components they
read -- both total-field calls read whatever value those two fields already held
*before this call to `Physics.run`*, not the value this same call is about to compute.
On the first pass through a run this reads `DataStructure`'s `0.0` default; on later
passes it reads the previous outer-loop (`Caller.call_models`, up to 10x
Gauss-Seidel) iteration's answer. Because
`b_plasma_inboard_toroidal`/`b_plasma_outboard_toroidal` depend only on
`b_plasma_toroidal_on_axis`, `rmajor`, `rminor` -- none of which this bug's own stale
read can perturb -- the stale and current values coincide once the outer loop has
converged, which is why `large_tokamak_eval.MFILE.DAT`'s converged
`b_plasma_outboard_total` (4.07616535601446195) exactly equals
`sqrt(b_plasma_outboard_toroidal**2 + b_plasma_surface_poloidal_average**2)` evaluated
at the *converged* `b_plasma_outboard_toroidal` (3.98874163098367829) -- see
`TotalMagneticFieldOutboard`'s docstring. **This port declares the correct edge (total
field reads the toroidal component computed in the same evaluation), not the stale
one** -- a cottax graph has no notion of "this call's read of a field it also writes
later," so there is no shape to reproduce the bug in; it dissolves into an ordinary DAG
edge. Flagged here rather than silently fixed, per this project's own convention for
undocumented PROCESS behaviour.

`physics.md` open question 1 asked for exactly this and declined to write it, on the
ground that `plasma_fields.py` was not that agent's file: *"Either that node is
generalised, or `plasma_fields.py` gains a second occupant. One line either way."* The
consolidation pass took the second occupant for the on-axis field; this pass adds its
inboard/outboard siblings the same way.

No harness contract of its own for the reused `calculate_total_field` call: the pure
function already has one (`tests/functional_process/models/stellarator/
test_plasma_physics.py`), and a second contract over the same function would diff
PROCESS against itself. The three toroidal-field/profile functions below are new pure
ports and do get their own contracts (`test_plasma_fields.py`).
"""

import dataclasses

import jax.numpy as jnp
from cottax.interfaces.pytree_namespace_module import (
    ExplicitFunction,
    From,
    ModelNamespace,
    OutputInto,
)

from functional_process.models.physics.physics import SurfaceAveragedPoloidalField
from functional_process.models.stellarator.plasma_physics import calculate_total_field
from functional_process.paths import physics

# ---------------------------------------------------------------------------
# Pure functions -- `process/models/physics/plasma_fields.py`'s remaining
# tokamak-relevant closure beyond `calculate_total_magnetic_field` (already reused
# above as `calculate_total_field`) and `calculate_surface_averaged_poloidal_field`
# (out of scope -- see module docstring).
# ---------------------------------------------------------------------------


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


class TotalMagneticField(ExplicitFunction):
    """cottax node: `physics.py:373-376`, the on-axis total field.

    Reads the toroidal field on axis and the surface-averaged poloidal field --  the
    latter being `SurfaceAveragedPoloidalFieldAmperes`' own output, so this node sits
    immediately downstream of its slot-mate and the edge between them is the only one
    inside `.tokamak.plasma_fields`.

    Unswitched. `calculate_total_magnetic_field` has no branch at all; which *arguments*
    it is called with is what `Physics.run` varies across its three call sites, and that
    is a call-site fact rather than a switch.
    """

    b_plasma_total = OutputInto(physics)

    def __call__(
        self,
        b_plasma_toroidal_on_axis=From(physics),
        b_plasma_surface_poloidal_average=From(physics),
    ):
        return calculate_total_field(
            b_plasma_toroidal_on_axis, b_plasma_surface_poloidal_average
        )


class PlasmaInboardToroidalField(ExplicitFunction):
    """`physics.py:394-401`, the inboard-midplane toroidal field component.

    Unswitched -- `calculate_plasma_inboard_toroidal_field` has no branch; which
    arguments `Physics.run` supplies is a call-site fact, not this function's, same
    reasoning as `TotalMagneticField`.
    """

    b_plasma_inboard_toroidal = OutputInto(physics)

    def __call__(
        self,
        b_plasma_toroidal_on_axis=From(physics),
        rmajor=From(physics),
        rminor=From(physics),
    ):
        return calculate_plasma_inboard_toroidal_field(
            b_plasma_toroidal_on_axis, rmajor, rminor
        )


class PlasmaOutboardToroidalField(ExplicitFunction):
    """`physics.py:403-409`, the outboard-midplane toroidal field component.

    Unswitched, same reasoning as `PlasmaInboardToroidalField`. Feeds
    `TotalMagneticFieldOutboard` below, whose two readers in `scrape_off_layer.py`
    are the reason this node exists in this pass.
    """

    b_plasma_outboard_toroidal = OutputInto(physics)

    def __call__(
        self,
        b_plasma_toroidal_on_axis=From(physics),
        rmajor=From(physics),
        rminor=From(physics),
    ):
        return calculate_plasma_outboard_toroidal_field(
            b_plasma_toroidal_on_axis, rmajor, rminor
        )


class TotalMagneticFieldInboard(ExplicitFunction):
    """`physics.py:386-392`, the inboard total field.

    Reads `PlasmaInboardToroidalField`'s own output -- **not** the value PROCESS itself
    reads at this call site, which is stale by one outer-loop pass; see the module
    docstring's "A genuine PROCESS ordering bug, not reproduced." No reader in the
    currently-assembled tokamak graph (`.physics.b_plasma_inboard_total` has none, only
    its outboard sibling does); wired anyway for symmetry with
    `TotalMagneticFieldOutboard`, at the same one-line cost `TotalMagneticField`'s own
    docstring already anticipated for both siblings.
    """

    b_plasma_inboard_total = OutputInto(physics)

    def __call__(
        self,
        b_plasma_inboard_toroidal=From(physics),
        b_plasma_surface_poloidal_average=From(physics),
    ):
        return calculate_total_field(
            b_plasma_inboard_toroidal, b_plasma_surface_poloidal_average
        )


class TotalMagneticFieldOutboard(ExplicitFunction):
    """`physics.py:378-384`, the outboard total field.

    **This is the node this pass exists to add.**
    `functional_process/models/physics/scrape_off_layer.py`'s
    `UpstreamSOLOutboardParallelArea` and `UpstreamSOLOutboardEich13ParallelArea` both
    declare a read of `.physics.b_plasma_outboard_total`
    (`scrape_off_layer.py:367,399`) with no producer anywhere in the ported graph before
    this class. Reads `PlasmaOutboardToroidalField`'s own output, not PROCESS's
    stale-by-one-pass read at this call site -- see the module docstring.

    Verified against `tests/regression/input_files/large_tokamak_eval.MFILE.DAT`'s
    converged operating point: `b_plasma_outboard_toroidal = 3.98874163098367829`,
    `b_plasma_surface_poloidal_average = 0.839681017309652056`, and
    `sqrt(3.98874163098367829**2 + 0.839681017309652056**2)` reproduces the reported
    `b_plasma_outboard_total = 4.07616535601446195` to float64 round-off -- the
    coincidence the module docstring's ordering-bug paragraph explains (stale and
    current values agree once the outer loop has converged).
    """

    b_plasma_outboard_total = OutputInto(physics)

    def __call__(
        self,
        b_plasma_outboard_toroidal=From(physics),
        b_plasma_surface_poloidal_average=From(physics),
    ):
        return calculate_total_field(
            b_plasma_outboard_toroidal, b_plasma_surface_poloidal_average
        )


class PlasmaFields(ModelNamespace):
    """`.tokamak.plasma_fields` -- the fields `PlasmaFields` produces for this graph.

    `process/models/physics/plasma_fields.py::PlasmaFields` is injected into `Physics`
    (`physics.py:197`) rather than run from `caller.py`, which is why it is a slot of
    `Tokamak` in its own right: `tokamak_call_surface.md` §A recorded it entered, and a
    sub-model reached by injection is a slot where §A named it.

    The stellarator's counterpart is `.stellarator.poloidal_field_from_rotational_
    transform` plus `.stellarator.total_field` -- the same two quantities from a
    rotational transform instead of a plasma current.
    """

    surface_averaged_poloidal_field: SurfaceAveragedPoloidalField = dataclasses.field(
        kw_only=True
    )
    """`.physics.i_plasma_current` -- the Ampere arm is written, the Peng arm
    (`i_plasma_current == 2`) calls `PlasmaCurrent.plascar_bpol` and is UNPORTED.
    Factory-filled; the annotation is the family base class."""

    total_magnetic_field: TotalMagneticField = TotalMagneticField()
    """`.physics.b_plasma_total`. Unswitched."""

    plasma_inboard_toroidal_field: PlasmaInboardToroidalField = (
        PlasmaInboardToroidalField()
    )
    """`.physics.b_plasma_inboard_toroidal`. Unswitched."""

    plasma_outboard_toroidal_field: PlasmaOutboardToroidalField = (
        PlasmaOutboardToroidalField()
    )
    """`.physics.b_plasma_outboard_toroidal`. Unswitched."""

    total_magnetic_field_inboard: TotalMagneticFieldInboard = TotalMagneticFieldInboard()
    """`.physics.b_plasma_inboard_total`. Unswitched. No reader yet in the ported
    graph."""

    total_magnetic_field_outboard: TotalMagneticFieldOutboard = (
        TotalMagneticFieldOutboard()
    )
    """`.physics.b_plasma_outboard_total`. Unswitched. Read by
    `scrape_off_layer.py`'s `UpstreamSOLOutboardParallelArea` and
    `UpstreamSOLOutboardEich13ParallelArea` -- this pass is what gives that read a
    producer."""
