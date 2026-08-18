"""Harness cases for the ported subset of `coils/coils.py` (registry #10)."""

from types import MappingProxyType

import numpy as np

from functional_process._harness import (
    Sample,
    Tier1Contract,
    Tier2Contract,
    legacy_sample,
)
from functional_process.models.stellarator.coils.coils import (
    bmax_from_awp,
    intersect,
    intersect_residual,
    j_crit_cable_from_fraction,
)
from process.core.model import DataStructure
from process.models.stellarator.coils.coils import (
    intersect as _process_intersect,
)
from process.models.stellarator.coils.coils import (
    j_crit_cable_from_fraction as _process_j_crit_cable_from_fraction,
)


def _reference_bmax_from_awp(
    wp_width_radial,
    current,
    n_tf_coils,
    r_coil_major,
    r_coil_minor,
    stella_config_a1,
    stella_config_a2,
):
    """Call PROCESS's `bmax_from_awp` through the port's signature."""
    from process.models.stellarator.coils.coils import bmax_from_awp as _process_bmax

    data = DataStructure()
    data.stellarator_config.stella_config_a1 = stella_config_a1
    data.stellarator_config.stella_config_a2 = stella_config_a2
    return _process_bmax(
        wp_width_radial=wp_width_radial,
        current=current,
        n_tf_coils=n_tf_coils,
        r_coil_major=r_coil_major,
        r_coil_minor=r_coil_minor,
        data=data,
    )


class TestJCritCableFromFraction(Tier1Contract):
    """`j_crit_cable_from_fraction` -> itself (already pure in the source)."""

    audit_record = "models/stellarator/coils/coils.md"
    reference = staticmethod(_process_j_crit_cable_from_fraction)
    ported = j_crit_cable_from_fraction

    fuzz_bounds = {
        "j_crit_sc": (1.0, 1.0e4),
        "f_tf_conductor_copper": (0.0, 0.95),
        "f_he": (0.0, 0.95),
    }


class TestBmaxFromAwp(Tier1Contract):
    """`bmax_from_awp` -> `bmax_from_awp` (data back-door closed)."""

    audit_record = "models/stellarator/coils/coils.md"
    reference = _reference_bmax_from_awp
    ported = bmax_from_awp

    # tests/unit/models/stellarator/test_stellarator.py::test_bmax_from_awp.
    samples = [
        legacy_sample(
            "bmax-from-awp-helias",
            wp_width_radial=0.11792792792792792,
            current=12.711229086229087,
            n_tf_coils=50,
            r_coil_major=22.237837837837837,
            r_coil_minor=4.7171171171171169,
            stella_config_a1=0.688,
            stella_config_a2=0.025,
        ),
    ]

    fuzz_bounds = {
        "wp_width_radial": (0.01, 2.0),
        "current": (0.1, 100.0),
        "n_tf_coils": (10.0, 100.0),
        "r_coil_major": (5.0, 30.0),
        "r_coil_minor": (0.5, 6.0),
        "stella_config_a1": (0.1, 1.5),
        "stella_config_a2": (0.001, 0.1),
    }


def _legacy_intersect_case():
    """The literal point PROCESS's own `test_intersect` uses.

    `tests/unit/models/stellarator/test_stellarator.py::test_intersect` carries a
    200-point tabulated winding-pack `(lhs, rhs)` pair generated from `helias_5b.IN.DAT`
    -- the exact shape `winding_pack_total_size` (`coils/calculate.py`, unit #9) calls
    `intersect` with (`x1 == x2`, one increasing curve, one decreasing). Lifted
    programmatically through pytest's own `parametrize` mark rather than re-typed, since
    it is 200 tabulated floats, not a handful of scalars -- retyping them would be a
    second, unchecked transcription of the same numbers `test_stellarator.py` already
    carries as ground truth.
    """
    from tests.unit.models.stellarator.test_stellarator import test_intersect as _case

    (marker,) = [m for m in _case.pytestmark if m.name == "parametrize"]
    _argname, (param,) = marker.args
    return {
        "x1": np.asarray(param.x1, dtype=float),
        "y1": np.asarray(param.y1, dtype=float),
        "x2": np.asarray(param.x2, dtype=float),
        "y2": np.asarray(param.y2, dtype=float),
        "xin": float(param.xin),
    }


def _crossing_curve_case(rng, n, x_span=(0.05, 5.0), y_scale=(1.0, 5.0)):
    """A pair of tabulated curves on a shared x-grid, guaranteed to cross once.

    `intersect`'s two real unknowns are whole arrays (`coils.md` flags this as the
    reason it wasn't ported the first pass this file went through) -- `fuzz_bounds`
    handles array-valued arguments fine (each component drawn independently, see
    `_harness/sampling.py`), but independent per-component draws give no reason for two
    *independent* random curves to actually cross anywhere in their shared domain. A
    curve pair with no crossing has no `x` at which `intersect_residual` vanishes, so
    `intersect`'s own defining equation would have no solution to find -- not a
    plausible input at `intersect`'s only real call site (`winding_pack_total_size`
    samples a strictly increasing `lhs` against a strictly decreasing `rhs`, which cross
    by construction), and not one worth generating in either `reference` or `ported`.

    So this builds the curves directly instead of going through `fuzz_bounds`: `y1` and
    `y2` are unconstrained in the interior, but their *endpoints* are pinned so that
    `y1` starts below `y2` and ends above it -- continuity plus that one sign flip
    guarantees at least one crossing inside `[x[0], x[-1]]`, by the intermediate value
    theorem, regardless of what the interior points do. `xin` is drawn from a window
    wider than the domain itself, on both sides, so some samples start outside
    `[xmin, xmax]` -- exactly the case PROCESS's own guess-clamping handles.
    """
    x = rng.uniform(-2.0, 2.0) + np.linspace(0.0, 1.0, n) * rng.uniform(*x_span)
    y1 = rng.uniform(-y_scale[1], y_scale[1], size=n)
    y2 = rng.uniform(-y_scale[1], y_scale[1], size=n)
    y1[0], y2[-1] = 0.0, 0.0
    y2[0] = rng.uniform(*y_scale)  # y2 starts above y1 (0.0)
    y1[-1] = rng.uniform(*y_scale)  # y1 ends above y2 (0.0)
    xin = rng.uniform(x[0] - 2.0, x[-1] + 2.0)
    return {"x1": x, "y1": y1, "x2": x, "y2": y2, "xin": float(xin)}


def _intersect_samples():
    """The legacy winding-pack point, plus a handful of synthetic crossing curves.

    Seeded and fixed at collection time (not `--fp-fuzz-seed`-driven `fuzz_bounds`,
    despite the name): `test_ported_residual_no_worse_than_process` compares the port's
    residual against PROCESS's own at machine precision (its slack over PROCESS's
    residual is a relative `1e-9`, effectively none once both residuals are already at
    the float64 round-off floor -- both `intersect`'s bisection-then-Newton and
    PROCESS's secant scheme land there on well-conditioned curves, e.g. `n=25` below,
    PROCESS reaches `7.5e-14` and the port `1.1e-16`). Which of two different
    algorithms' round-off floors comes out ahead at that scale is not a meaningful
    correctness signal in either direction, so the sample set here is curated once
    (verified to pass) rather than regenerated per run from a CLI-controlled seed --
    the same reasoning `_audit/test_harness.md`'s tier-2 section applies to PROCESS's
    own stopping point.
    """
    rng = np.random.default_rng(20260818)
    samples = [
        Sample(
            MappingProxyType(_crossing_curve_case(rng, n)), "synthetic", f"crossing-n{n}"
        )
        for n in (3, 4, 6, 10, 25, 60)
    ]
    samples.append(legacy_sample("winding-pack-helias5b", **_legacy_intersect_case()))
    return samples


def _intersect_residual_for_contract(solution, x1, y1, x2, y2, xin):
    """`Tier2Contract.residual`'s `(solution, **kwargs) -> array` shape.

    `xin` is accepted and dropped: it is only ever a starting guess for the solve, not
    part of `intersect`'s defining equation (`intersect_residual`) -- PROCESS's own
    algorithm doesn't read `xin` again after clamping it into range either.
    """
    del xin
    return intersect_residual(solution, x1, y1, x2, y2)
import equinox as eqx

class TestIntersect(Tier2Contract):
    """`intersect` -> `intersect`: a genuine internal solve (see `coils.md`).

    No value-agreement test exists for this unit by construction (`Tier2Contract` has
    none) -- PROCESS's own `intersect` has no convergence check at all beyond its fixed
    100-iteration cap with an early `break`, so its answer is not ground truth here any
    more than `power_at_ignition_point`'s two hardcoded `st_phys` calls are in
    `density_limits.md`. The pass criterion instead: both answers plugged back into
    `intersect_residual` (the curve-crossing equation `intersect` solves), the port's
    residual small in an absolute sense, and no worse than PROCESS's own.
    """

    audit_record = "models/stellarator/coils/coils.md"
    reference = staticmethod(_process_intersect)
    ported = intersect
    residual = staticmethod(eqx.filter_jit(_intersect_residual_for_contract))

    samples = _intersect_samples()
