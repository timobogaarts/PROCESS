"""Harness cases for the ported subset of `coils/coils.py` (registry #10)."""

from types import MappingProxyType

import jax.numpy as jnp
import numpy as np
import pytest
from cottax.tools.path import path_map
from cottax.blocking import Blocking
from cottax.evaluate import Schedule
from cottax.interfaces.pytree_namespace_module import resolve, to_graph
from cottax.problem import RootFind, Start, driver_vars
from cottax.rewrites import Assign
from cottax.spec import VarPath

from functional_process._harness import (
    Sample,
    Tier1Contract,
    Tier2Contract,
    legacy_sample,
)
from functional_process.models.stellarator.coils.coils import (
    Intersect,
    IntersectBisectionNewtonPolish,
    bmax_from_awp,
    intersect,
    intersect_residual,
    j_crit_cable_from_fraction,
    jcrit_from_material_bi2212,
    jcrit_from_material_gl_nbti,
    jcrit_from_material_gl_rebco,
    jcrit_from_material_iter_nb3sn,
    jcrit_from_material_iter_nb3sn_user_defined,
    jcrit_from_material_nbti_lubell,
    jcrit_from_material_rebco,
    jcrit_from_material_wst_nb3sn,
)
from functional_process.paths import stellarator
from process.core.model import DataStructure
from process.models import superconductors as _process_superconductors
from process.models.stellarator.coils.coils import (
    intersect as _process_intersect,
)
from process.models.stellarator.coils.coils import (
    j_crit_cable_from_fraction as _process_j_crit_cable_from_fraction,
)
from process.models.stellarator.coils.coils import (
    jcrit_from_material as _process_jcrit_from_material,
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


def _reference_jcrit_branch(i_tf_sc_mat, **overrides):
    """Call PROCESS's real `jcrit_from_material` dispatcher, one branch fixed.

    `jcrit_from_material` takes all 11 of its arguments regardless of which
    `i_tf_sc_mat` branch actually reads them (`coils.md`/`superconductors.md`'s
    per-branch reads-set table) -- unused slots are filled with `0.0` here, since the
    branch under test never reads them (confirmed by the reads-set audit, not assumed).
    """
    kwargs = dict(
        b_crit_upper_nbti=0.0,
        b_crit_sc=0.0,
        f_a_tf_turn_cable_copper=0.0,
        f_hts=0.0,
        t_crit_nbti=0.0,
        t_crit_sc=0.0,
        f_a_tf_turn_cable_space_extra_void=0.0,
        j_wp=0.0,
    )
    kwargs.update(overrides)
    return _process_jcrit_from_material(i_tf_sc_mat=i_tf_sc_mat, **kwargs)


def _reference_jcrit_iter_nb3sn(t_helium, b_max):
    return _reference_jcrit_branch(1, b_max=b_max, t_helium=t_helium)


def _reference_jcrit_nbti_lubell(t_helium, b_max):
    return _reference_jcrit_branch(3, b_max=b_max, t_helium=t_helium)


def _reference_jcrit_iter_nb3sn_user_defined(t_helium, b_max, bcritsc, tcritsc):
    return _reference_jcrit_branch(
        4, b_max=b_max, t_helium=t_helium, b_crit_sc=bcritsc, t_crit_sc=tcritsc
    )


def _reference_jcrit_wst_nb3sn(t_helium, b_max):
    return _reference_jcrit_branch(5, b_max=b_max, t_helium=t_helium)


def _reference_jcrit_gl_nbti(t_helium, b_max, b_crit_upper_nbti, t_crit_nbti):
    return _reference_jcrit_branch(
        7,
        b_max=b_max,
        t_helium=t_helium,
        b_crit_upper_nbti=b_crit_upper_nbti,
        t_crit_nbti=t_crit_nbti,
    )


def _reference_jcrit_gl_rebco(t_helium, b_max):
    return _reference_jcrit_branch(8, b_max=b_max, t_helium=t_helium)


def _reference_jcrit_bi2212(
    t_helium,
    b_max,
    j_tf_wp,
    f_a_tf_turn_cable_space_extra_void,
    fhts,
    f_a_tf_turn_cable_copper,
):
    return _reference_jcrit_branch(
        2,
        b_max=b_max,
        t_helium=t_helium,
        j_wp=j_tf_wp,
        f_a_tf_turn_cable_space_extra_void=f_a_tf_turn_cable_space_extra_void,
        f_hts=fhts,
        f_a_tf_turn_cable_copper=f_a_tf_turn_cable_copper,
    )


def _reference_jcrit_rebco(t_helium, b_max):
    """PROCESS's real `i_tf_sc_mat == 6` call site is unreachable, not a ground truth.

    `jcrit_from_material`'s own `i_tf_sc_mat == 6` branch calls
    `superconductors.jcrit_rebco(t_helium, b_max, 0)` -- three positional arguments to a
    two-argument function -- so it raises `TypeError` unconditionally, confirmed directly
    (see `coils.md`'s open question / `superconductors.md`'s open question 1). There is no
    PROCESS answer to treat as ground truth for this branch, so this reference instead
    calls `superconductors.jcrit_rebco` correctly (the same fix
    `jcrit_from_material_rebco`/`calculate.py`'s `_critical_current_density_by_material`
    already apply) and reproduces the rest of the branch's real post-processing (the
    `max(1e-9, ...)` floor, the `1e-6` scaling) by hand -- this is what the port's own
    branch does, so this is the correct oracle for it, not the (unreachable) dispatcher.
    """
    j_crit_sc, _validity, _b_c20max, _temp_c0max = _process_superconductors.jcrit_rebco(
        t_helium, b_max
    )
    return max(1.0e-9, j_crit_sc) * 1.0e-6


class TestJcritFromMaterialIterNb3sn(Tier1Contract):
    """`jcrit_from_material`, `i_tf_sc_mat == 1` (ITER Nb3Sn) -> `jcrit_from_material_iter_nb3sn`."""

    audit_record = "models/stellarator/coils/coils.md"
    reference = staticmethod(_reference_jcrit_iter_nb3sn)
    ported = jcrit_from_material_iter_nb3sn

    samples = [
        Sample(
            MappingProxyType({"t_helium": 4.75, "b_max": 10.0}),
            "synthetic",
            "below-bc20m",
        ),
        Sample(
            MappingProxyType({"t_helium": 4.75, "b_max": 40.0}),
            "synthetic",
            "above-bc20m",
        ),
    ]

    fuzz_bounds = {
        "t_helium": (1.0, 15.5),
        "b_max": (0.5, 40.0),
    }


class TestJcritFromMaterialBi2212(Tier1Contract):
    """`jcrit_from_material`, `i_tf_sc_mat == 2` (Bi-2212) -> `jcrit_from_material_bi2212`."""

    audit_record = "models/stellarator/coils/coils.md"
    reference = staticmethod(_reference_jcrit_bi2212)
    ported = jcrit_from_material_bi2212

    samples = [
        Sample(
            MappingProxyType({
                "t_helium": 4.75,
                "b_max": 8.0,
                "j_tf_wp": 5.0e5,
                "f_a_tf_turn_cable_space_extra_void": 0.3,
                "fhts": 0.5,
                "f_a_tf_turn_cable_copper": 0.4,
            }),
            "synthetic",
            "in-range",
        ),
    ]

    fuzz_bounds = {
        "t_helium": (1.0, 10.0),
        "b_max": (6.5, 20.0),
        "j_tf_wp": (1.0e5, 3.0e7),
        "f_a_tf_turn_cable_space_extra_void": (0.0, 0.6),
        "fhts": (0.1, 1.0),
        "f_a_tf_turn_cable_copper": (0.0, 0.6),
    }


class TestJcritFromMaterialNbtiLubell(Tier1Contract):
    """`jcrit_from_material`, `i_tf_sc_mat == 3` (NbTi, Lubell) -> `jcrit_from_material_nbti_lubell`."""

    audit_record = "models/stellarator/coils/coils.md"
    reference = staticmethod(_reference_jcrit_nbti_lubell)
    ported = jcrit_from_material_nbti_lubell

    samples = [
        Sample(
            MappingProxyType({"t_helium": 4.75, "b_max": 8.0}),
            "synthetic",
            "below-bc20m",
        ),
        Sample(
            MappingProxyType({"t_helium": 4.75, "b_max": 20.0}),
            "synthetic",
            "above-bc20m",
        ),
    ]

    fuzz_bounds = {
        "t_helium": (1.0, 9.0),
        "b_max": (0.5, 20.0),
    }


class TestJcritFromMaterialIterNb3snUserDefined(Tier1Contract):
    """`jcrit_from_material`, `i_tf_sc_mat == 4` -> `jcrit_from_material_iter_nb3sn_user_defined`."""

    audit_record = "models/stellarator/coils/coils.md"
    reference = staticmethod(_reference_jcrit_iter_nb3sn_user_defined)
    ported = jcrit_from_material_iter_nb3sn_user_defined

    samples = [
        Sample(
            MappingProxyType({
                "t_helium": 4.75,
                "b_max": 15.0,
                "bcritsc": 22.0,
                "tcritsc": 12.0,
            }),
            "synthetic",
            "user-defined",
        ),
    ]

    fuzz_bounds = {
        "t_helium": (1.0, 15.5),
        "b_max": (0.5, 30.0),
        "bcritsc": (25.0, 35.0),
        "tcritsc": (14.0, 18.0),
    }


class TestJcritFromMaterialWstNb3sn(Tier1Contract):
    """`jcrit_from_material`, `i_tf_sc_mat == 5` (WST Nb3Sn) -> `jcrit_from_material_wst_nb3sn`."""

    audit_record = "models/stellarator/coils/coils.md"
    reference = staticmethod(_reference_jcrit_wst_nb3sn)
    ported = jcrit_from_material_wst_nb3sn

    samples = [
        Sample(
            MappingProxyType({"t_helium": 4.75, "b_max": 20.0}),
            "synthetic",
            "in-range",
        ),
    ]

    fuzz_bounds = {
        "t_helium": (1.0, 15.0),
        "b_max": (0.5, 26.0),
    }


class TestJcritFromMaterialRebco(Tier1Contract):
    """`jcrit_from_material`, `i_tf_sc_mat == 6` (REBCO) -> `jcrit_from_material_rebco`.

    See `_reference_jcrit_rebco`'s docstring: PROCESS's own dispatcher raises
    unconditionally for this branch (a real bug, not reproduced), so the reference here
    calls `superconductors.jcrit_rebco` correctly instead of going through
    `jcrit_from_material`.
    """

    audit_record = "models/stellarator/coils/coils.md"
    reference = staticmethod(_reference_jcrit_rebco)
    ported = jcrit_from_material_rebco

    samples = [
        Sample(
            MappingProxyType({"t_helium": 4.75, "b_max": 10.0}),
            "synthetic",
            "in-range",
        ),
    ]

    fuzz_bounds = {
        "t_helium": (1.0, 60.0),
        "b_max": (0.5, 14.0),
    }


class TestJcritFromMaterialGlNbti(Tier1Contract):
    """`jcrit_from_material`, `i_tf_sc_mat == 7` (Durham GL Nb-Ti) -> `jcrit_from_material_gl_nbti`."""

    audit_record = "models/stellarator/coils/coils.md"
    reference = staticmethod(_reference_jcrit_gl_nbti)
    ported = jcrit_from_material_gl_nbti

    samples = [
        Sample(
            MappingProxyType({
                "t_helium": 4.75,
                "b_max": 6.0,
                "b_crit_upper_nbti": 14.0,
                "t_crit_nbti": 8.5,
            }),
            "synthetic",
            "in-range",
        ),
    ]

    fuzz_bounds = {
        "t_helium": (1.0, 9.0),
        "b_max": (0.5, 12.0),
        "b_crit_upper_nbti": (7.0, 12.0),
        "t_crit_nbti": (10.0, 16.0),
    }


class TestJcritFromMaterialGlRebco(Tier1Contract):
    """`jcrit_from_material`, `i_tf_sc_mat == 8` (Durham GL REBCO) -> `jcrit_from_material_gl_rebco`."""

    audit_record = "models/stellarator/coils/coils.md"
    reference = staticmethod(_reference_jcrit_gl_rebco)
    ported = jcrit_from_material_gl_rebco

    samples = [
        Sample(
            MappingProxyType({"t_helium": 4.75, "b_max": 15.0}),
            "synthetic",
            "in-range",
        ),
    ]

    fuzz_bounds = {
        "t_helium": (1.0, 9.0),
        "b_max": (0.5, 30.0),
    }


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


# ---------------------------------------------------------------------------
# `Intersect` -- `intersect` as a genuine `ImplicitFunction`/`RootFind` pair, plus a
# concrete, test-only `AbstractDriver` (`IntersectBisectionNewtonPolish`) wrapping
# `intersect`'s own bisection-then-Newton-polish algorithm exactly.
#
# `next_steps.md` §7 already established that nothing else in the graph needs
# `intersect`'s internal unknowns visible -- that conclusion is unchanged here. The
# reason to declare `Intersect` anyway: the solver algorithm becomes a first-class,
# swappable `Drive` choice instead of something hardcoded inside `intersect`'s own body.
# These checks are the actual point: prove the structure assembles and that a concrete
# driver still recovers exactly `intersect`'s own answer, not just assert it.
# ---------------------------------------------------------------------------


def test_intersect_declares_a_body_and_a_root_find_problem():
    """`to_graph(Intersect())` must assemble cleanly into exactly two nodes: the
    `residual` body (a `ImplementedFunction`) and the `RootFind` problem it feeds -- the pair
    `ImplicitFunction.node_definitions_and_names` documents.
    """
    node = Intersect()
    graph = to_graph(node)
    assert len(graph.definitions) == 2
    assert not graph.is_acyclic
    assert graph.declared == (node.problem_name,)
    assert graph.problem_type is RootFind


def test_intersect_body_reads_the_unknown_back_without_owning_it():
    """`residual` reads `.stellarator.wp_width_r_min` (the same `VarPath` its own
    `Output` declares) alongside the two curves -- not a self-loop: the body node only
    *reads* the real path and *writes* the minted `^cond` copy; the separate `RootFind`
    problem node is the one that actually owns `.stellarator.wp_width_r_min`.
    """
    node = Intersect()
    body, problem = node.node_definitions
    unknown = resolve(stellarator.wp_width_r_min, VarPath)
    assert unknown in body.reads
    assert unknown not in body.owns
    assert problem.owns == (unknown,)
    assert problem.reads == body.owns


def test_intersect_has_no_port_for_xin():
    """`intersect`'s `xin` argument has no place in `Intersect`'s declared interface at
    all -- `residual` declares exactly 4 `FromExactly`s (the unknown plus the two curves), one
    fewer than `intersect`'s own 5-argument signature -- confirming `coils.md`'s open
    question 2 directly rather than merely asserting it.
    """
    node = Intersect()
    assert len(node.inputs) == 4
    assert intersect.__code__.co_argcount == 5


def test_intersect_bisection_newton_polish_drives_to_the_same_answer_as_intersect():
    """The actual numeric point: driving `Intersect`'s declared `RootFind` with
    `IntersectBisectionNewtonPolish` (test-only, wraps `intersect` exactly) must recover
    precisely the value `intersect` itself computes on the same curves -- same algorithm,
    reached through the new structural/driven path instead of a plain eager call.

    Reuses `_intersect_samples()` (the same curated crossing-curve samples
    `TestIntersect` itself checks against PROCESS) rather than inventing a new fixture --
    this is checking the *port's own* two ways of getting an answer agree, not a fresh
    correctness claim about `intersect` itself.
    """
    node = Intersect()
    # `Initialise` declares the problem's `Start` port, which is where the driver now
    # reads its guess from; without it `Drive` refuses the pair, and seeding the
    # unknown's own name would silently fall back to the median instead (a different
    # root -- see the `env` comment below).
    graph = Assign(node.problem_name, IntersectBisectionNewtonPolish()).apply(
        to_graph(node)
    )
    (guess_path,) = driver_vars(graph[node.problem_name], Start)
    schedule = Schedule(Blocking.scc(graph))
    wp_width_r_path = resolve(stellarator.wp_width_r, VarPath)
    lhs_path = resolve(stellarator.lhs, VarPath)
    rhs_path = resolve(stellarator.rhs, VarPath)
    wp_width_r_min_path = resolve(stellarator.wp_width_r_min, VarPath)

    for sample in _intersect_samples():
        kwargs = sample.kwargs
        env = {
            wp_width_r_path: jnp.asarray(kwargs["x1"]),
            lhs_path: jnp.asarray(kwargs["y1"]),
            rhs_path: jnp.asarray(kwargs["y2"]),
            # Seed the same starting guess `intersect` itself was called with below,
            # at the problem's `^guess.*` port -- a start is declared driver data now,
            # not an already-present unknown. Some `_crossing_curve_case` samples have
            # more than one genuine crossing in-domain (only the *sign* of the endpoints
            # is pinned, not the interior), so bisection's own answer can genuinely
            # depend on where it starts -- matching `xin` here is what makes this an
            # apples-to-apples check of "same algorithm, driven instead of eager", not a
            # claim that any starting guess reaches the same root.
            guess_path: jnp.asarray(kwargs["xin"]),
        }
        out = schedule.run(path_map(env))
        want = intersect(
            jnp.asarray(kwargs["x1"]),
            jnp.asarray(kwargs["y1"]),
            jnp.asarray(kwargs["x2"]),
            jnp.asarray(kwargs["y2"]),
            kwargs["xin"],
        )
        assert out[wp_width_r_min_path] == pytest.approx(float(want)), sample.label
