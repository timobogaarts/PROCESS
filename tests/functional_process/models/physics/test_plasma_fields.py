"""Harness cases for the ported subset of `process/models/physics/plasma_fields.py`
(`PlasmaFields`, `.tokamak.plasma_fields`).

Audit record: `functional_process/_audit/units/models/physics/plasma_fields.md`. Read it
first for the "already ported" section (this file's `TotalMagneticField` node and the
imported `calculate_total_field` formula predate this pass and are not re-tested here --
that node has its own harness-free rationale documented at the top of the port module)
and for "a genuine PROCESS ordering bug" (why `TotalMagneticFieldInboard`/`Outboard`
deliberately do *not* reproduce `Physics.run()`'s one-call-stale read).

No `tests/unit/models/physics/test_plasma_fields.py` exists in `process/` to lift a
legacy sample from -- checked, the file is absent. Legacy samples here instead call
PROCESS's own `@staticmethod`s directly, in-process, at the converged operating point
recorded in `tests/regression/input_files/large_tokamak_eval.MFILE.DAT`
(`rmajor = 8.0`, `rminor = 2.66666666666666652`,
`b_plasma_toroidal_on_axis = 5.31832217464490409`,
`b_plasma_surface_poloidal_average = 0.839681017309652056`) -- genuinely "legacy" in the
harness's sense (a real, already-validated PROCESS answer), just sourced from a
regression MFILE rather than a `tests/unit` parametrisation.
"""

from functional_process._harness import Tier1Contract, legacy_sample
from functional_process.cottax.physics.plasma_fields import (
    PlasmaInboardToroidalField,
    PlasmaOutboardToroidalField,
    TotalMagneticFieldInboard,
    TotalMagneticFieldOutboard,
    calculate_plasma_inboard_toroidal_field,
    calculate_plasma_outboard_toroidal_field,
    calculate_toroidal_field_profile,
)
from process.models.physics.plasma_fields import PlasmaFields


class TestCalculatePlasmaInboardToroidalField(Tier1Contract):
    """`calculate_plasma_inboard_toroidal_field` -> the same, unchanged.

    Reference call: `PlasmaFields.calculate_plasma_inboard_toroidal_field`, PROCESS's
    own `@staticmethod`/`@nb.njit`, called in-process (no `self.data` access, so no
    `DataStructure` adapter is needed).
    """

    audit_record = "models/physics/plasma_fields.md"
    reference = staticmethod(PlasmaFields.calculate_plasma_inboard_toroidal_field)
    ported = calculate_plasma_inboard_toroidal_field

    samples = [
        legacy_sample(
            "large_tokamak_eval-converged",
            b_plasma_toroidal_on_axis=5.31832217464490409,
            rmajor=8.0,
            rminor=2.66666666666666652,
        ),
    ]

    fuzz_bounds = {
        "b_plasma_toroidal_on_axis": (1.0, 12.0),
        "rmajor": (2.0, 20.0),
        "rminor": (0.5, 1.9),
    }
    """`rminor` capped below `rmajor`'s own lower bound (2.0) so fuzz never approaches
    the `rmajor == rminor` singularity documented in the audit record's JAX-difficulty
    flags -- the sampler draws `rmajor` and `rminor` independently, so nothing else
    keeps them apart."""


class TestCalculatePlasmaOutboardToroidalField(Tier1Contract):
    """`calculate_plasma_outboard_toroidal_field` -> the same, unchanged. No
    singularity on the physical domain (a sum, not a difference, of two positive
    lengths).
    """

    audit_record = "models/physics/plasma_fields.md"
    reference = staticmethod(PlasmaFields.calculate_plasma_outboard_toroidal_field)
    ported = calculate_plasma_outboard_toroidal_field

    samples = [
        legacy_sample(
            "large_tokamak_eval-converged",
            b_plasma_toroidal_on_axis=5.31832217464490409,
            rmajor=8.0,
            rminor=2.66666666666666652,
        ),
    ]

    fuzz_bounds = {
        "b_plasma_toroidal_on_axis": (1.0, 12.0),
        "rmajor": (2.0, 20.0),
        "rminor": (0.5, 5.0),
    }


class TestCalculateToroidalFieldProfile(Tier1Contract):
    """`calculate_toroidal_field_profile` -> the same, `np.` -> `jnp.` only.

    `n_plasma_profile_elements` is a static array-shape count, not a continuous
    physics input (audit record's JAX-difficulty flags) -- held fixed across both the
    legacy sample and every fuzz point, and excluded from differentiation via
    `static_argnames` so `--fp-gradients` never attempts a `jacfwd` column for it. Held
    at `3` (not PROCESS's own default `201`) purely for harness speed: the function's
    correctness does not depend on the count, and a smaller array exercises exactly the
    same code path (linspace + zero-guard + broadcast division) as the default.
    """

    audit_record = "models/physics/plasma_fields.md"
    reference = staticmethod(PlasmaFields.calculate_toroidal_field_profile)
    ported = calculate_toroidal_field_profile

    static_argnames = ("n_plasma_profile_elements",)

    samples = [
        legacy_sample(
            "large_tokamak_eval-converged-n3",
            b_plasma_toroidal_on_axis=5.31832217464490409,
            rmajor=8.0,
            rminor=2.66666666666666652,
            n_plasma_profile_elements=3,
        ),
    ]

    fuzz_fixed = {"n_plasma_profile_elements": 3}
    fuzz_bounds = {
        "b_plasma_toroidal_on_axis": (1.0, 12.0),
        "rmajor": (2.0, 20.0),
        "rminor": (0.5, 1.9),
    }
    """Same `rminor` cap as `TestCalculatePlasmaInboardToroidalField`: at the magnetic
    axis (`rho == 0`, i.e. `rmajor == rminor`) PROCESS's own zero-guard
    (`jnp.where(rho == 0, 1e-10, rho)`) already gives a well-defined value, but nothing
    in this test exercises that guard on purpose -- it is a documented feature of the
    ported function, not something this contract is checking."""


def test_inboard_toroidal_field_node_reads_and_writes():
    """`PlasmaInboardToroidalField` reads exactly `b_plasma_toroidal_on_axis`,
    `rmajor`, `rminor` and owns exactly `.physics.b_plasma_inboard_toroidal` --
    pinned so a future edit cannot silently rebind either side.
    """
    node = PlasmaInboardToroidalField()
    reads = {i.var.path_str() for i in node.inputs}
    writes = {o.var.path_str() for o in node.outputs}
    assert reads == {
        ".physics.b_plasma_toroidal_on_axis",
        ".physics.rmajor",
        ".physics.rminor",
    }
    assert writes == {".physics.b_plasma_inboard_toroidal"}


def test_outboard_toroidal_field_node_reads_and_writes():
    """`PlasmaOutboardToroidalField`'s counterpart to the test above."""
    node = PlasmaOutboardToroidalField()
    reads = {i.var.path_str() for i in node.inputs}
    writes = {o.var.path_str() for o in node.outputs}
    assert reads == {
        ".physics.b_plasma_toroidal_on_axis",
        ".physics.rmajor",
        ".physics.rminor",
    }
    assert writes == {".physics.b_plasma_outboard_toroidal"}


def test_total_magnetic_field_inboard_reads_the_inboard_toroidal_component():
    """`TotalMagneticFieldInboard` reads `PlasmaInboardToroidalField`'s own output
    (`.physics.b_plasma_inboard_toroidal`), not the on-axis or outboard toroidal
    field -- the wiring mistake the ordering-bug discussion in the audit record makes
    easy to make by accident (three near-identical total-field call sites in
    `physics.py`, one toroidal-field argument differing between them).
    """
    node = TotalMagneticFieldInboard()
    reads = {i.var.path_str() for i in node.inputs}
    writes = {o.var.path_str() for o in node.outputs}
    assert reads == {
        ".physics.b_plasma_inboard_toroidal",
        ".physics.b_plasma_surface_poloidal_average",
    }
    assert writes == {".physics.b_plasma_inboard_total"}


def test_total_magnetic_field_outboard_reads_the_outboard_toroidal_component():
    """`TotalMagneticFieldOutboard`'s counterpart to the test above -- the node this
    pass exists to add, since `scrape_off_layer.py`'s `UpstreamSOLOutboardParallelArea`
    and `UpstreamSOLOutboardEich13ParallelArea` both declare a read of
    `.physics.b_plasma_outboard_total` with no other producer.
    """
    node = TotalMagneticFieldOutboard()
    reads = {i.var.path_str() for i in node.inputs}
    writes = {o.var.path_str() for o in node.outputs}
    assert reads == {
        ".physics.b_plasma_outboard_toroidal",
        ".physics.b_plasma_surface_poloidal_average",
    }
    assert writes == {".physics.b_plasma_outboard_total"}
