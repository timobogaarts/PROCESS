"""`architectures.stages` on `stellarator_helias`'s driven MDA graph.

One graph now (152 nodes), not two. `df050df3` removed three declared
`FixedPointFunction` nodes, and `4147b6bc` unregistered the unconsumed
`.vacuum.duct_diameter_root_find` island -- nothing produced what it read or read
what it produced -- which emptied `evaluate.EXCLUDED_NODE_NAMES` and made
`without_excluded` a no-op. So the two fixtures below build byte-identical graphs;
what still distinguishes the two parametrized tests that use them is the **claim
set** `violations` is checked against, not the graph: `tabled` checks `TABLED_CLAIMS`
(sections 1a-1d of `output_kinds.md`, the claims the handoff's stage check had); `full`
checks the whole of `CLAIMED_BUILD_OUTPUTS`, including the two section-4 lifetimes
`kinds.py` added after the handoff's check. The three rows of the handoff's table
(`~/jaxgraph`, `plans/handoff_2026-09-17.md`), against `tabled`'s claim set:

| set     | leaves   | first-stage nodes | claimed-build violations |
|---------|----------|-------------------|--------------------------|
| all     | 146 + 16 | 25 / 152          | 13                       |
| sampled | 26 + 2   | 34 / 152          | 13                       |
| build   | 22 + 2   | 58 / 152          | 4                        |

(The handoff's own numbers -- 27/156, 36/156, 60/156 -- were measured on a since-shrunk
graph; the two deleted nodes were both first stage, hence 27 -> 25, 36 -> 34, 60 -> 58.)
Against the full claim set, each row picks up two more violations -- the two
section-4 lifetimes, both `Decision.RECOURSE`.

The assertion that matters is the `build` row's four accepted violations.
"""

import jax
import numpy as np
import pytest

from functional_process.configurations import load
from functional_process.configurations.kinds import (
    BUILD_LEAVES,
    CLAIMED_BUILD_OUTPUTS,
    SIZING_CHOICES,
    Decision,
)
from functional_process.cottax.architectures import mda
from functional_process.cottax.architectures.evaluate import (
    mda_schedule,
    run_schedule,
    seed_env,
    without_excluded,
)
from functional_process.cottax.architectures.stages import (
    RECOURSE_PATHS,
    Stage,
    first_stage_graph,
    hoisted,
    hoisted_inputs,
    leaves,
    recourse_graph,
    report,
    sampled_paths,
    split,
    violations,
)
from functional_process.cottax.input import native
from functional_process.cottax.input.indat import graph_for

jax.config.update("jax_enable_x64", True)

ACCEPTED = (
    ".fwbs.life_fw_fpy",
    ".heat_transport.n_primary_heat_exchangers",
    ".vacuum.dia_vv_vacuum_ducts",
    ".vacuum.n_vac_pumps_high",
)
"""The four second-stage build outputs the 2026-09-17 decisions accept."""

LIFETIMES = (".costs.life_div_fpy", ".fwbs.life_blkt_fpy")
"""The two section-4 claims `kinds.py` added after the handoff's check."""

TABLED_CLAIMS = {s: sec for s, sec in CLAIMED_BUILD_OUTPUTS.items() if sec != "4"}
"""The claims the handoff's check had: sections 1a-1d of `output_kinds.md`."""


@pytest.fixture(scope="module")
def configuration():
    """The configuration the handoff measured."""
    return load("stellarator_helias")


@pytest.fixture(scope="module")
def full(configuration):
    """The driven MDA graph, checked against the full claim set: 152 nodes."""
    return mda.driven_graph(without_excluded(graph_for(configuration.machine)))


@pytest.fixture(scope="module")
def tabled(configuration):
    """The same driven MDA graph, checked against `TABLED_CLAIMS`: 152 nodes."""
    return mda.driven_graph(graph_for(configuration.machine))


def rows(graph):
    """The three leaf sets of the handoff's table, on `graph`."""
    return {
        "all": leaves(graph),
        "sampled": leaves(graph, sampled=sampled_paths(), varying=RECOURSE_PATHS),
        "build": leaves(
            graph, sampled=sampled_paths(held=BUILD_LEAVES), varying=RECOURSE_PATHS
        ),
    }


def test_leaves_are_the_handoffs(full):
    """The three leaf sets have the handoff's sizes."""
    sets = rows(full)
    assert (len(sets["all"].belief), len(sets["all"].operating)) == (146, 16)
    assert (len(sets["sampled"].belief), len(sets["sampled"].operating)) == (26, 2)
    assert (len(sets["build"].belief), len(sets["build"].operating)) == (22, 2)
    assert set(sets["build"].belief) == set(sets["sampled"].belief) - {
        v for v in sets["sampled"].belief if v.spelling in BUILD_LEAVES
    }
    assert all(v.spelling not in BUILD_LEAVES for v in sets["build"].belief)


@pytest.mark.parametrize(
    ("label", "first", "n_violations"),
    [("all", 25, 13), ("sampled", 34, 13), ("build", 58, 4)],
)
def test_handoff_table_reproduces(tabled, label, first, n_violations):
    """The handoff's row, on the graph and with the claims it was measured with."""
    assert len(tabled.nodes) == 152
    stages = split(tabled, rows(tabled)[label])
    hits, _clean = violations(tabled, stages, TABLED_CLAIMS)
    assert (len(stages.first), len(hits)) == (first, n_violations), report(
        stages, hits, label
    )


@pytest.mark.parametrize(
    ("label", "first", "n_violations"),
    [("all", 25, 15), ("sampled", 34, 15), ("build", 58, 6)],
)
def test_full_claims_rows(full, label, first, n_violations):
    """The same rows, same graph, against the full claim set: two more violations
    per row -- the two section-4 lifetimes.
    """
    assert len(full.nodes) == 152
    stages = split(full, rows(full)[label])
    assert stages.n_nodes == 152
    assert sum(c.nodes for c in stages.counts.values()) == 152
    hits, clean = violations(full, stages)
    assert (len(stages.first), len(hits)) == (first, n_violations), report(
        stages, hits, label
    )
    assert len(hits) + len(clean) == len([
        s
        for s in CLAIMED_BUILD_OUTPUTS
        if s in {v.spelling for v in full.graph.owners}
    ])
    # Sorted by how many leaves are responsible, most first.
    assert [v.n_responsible for v in hits] == sorted(
        (v.n_responsible for v in hits), reverse=True
    )


def test_build_row_leaves_exactly_the_accepted_violations(full):
    """With `BUILD_LEAVES` at nominal, the four accepted outputs are what still varies
    -- plus the two lifetimes `SIZING_CHOICES` reclassifies as recourse.
    """
    stages = split(full, rows(full)["build"])
    hits, clean = violations(full, stages, TABLED_CLAIMS)
    assert tuple(sorted(v.place for v in hits)) == ACCEPTED
    hits_all, _ = violations(full, stages)
    assert tuple(sorted(v.place for v in hits_all)) == tuple(
        sorted(ACCEPTED + LIFETIMES)
    )
    assert all(SIZING_CHOICES[v.place] is not Decision.LIFT for v in hits_all)
    assert all(SIZING_CHOICES[p] is Decision.RECOURSE for p in LIFETIMES)
    # Every coil, radial-build and first-wall claim is first stage once the four
    # build leaves are held.
    confirmed = {c.place for c in clean}
    assert {
        ".stellarator.wp_width_r_min",
        ".tfcoil.dr_tf_wp_with_insulation",
        ".tfcoil.j_tf_wp",
        ".build.dr_fw_inboard",
        ".build.dr_tf_outboard",
    } <= confirmed
    # A violation names the leaves that reach it, and none is a build leaf.
    for v in hits_all:
        assert v.responsible
        assert not set(v.responsible) & set(BUILD_LEAVES)
        assert v.stage in {Stage.SECOND, Stage.RECOURSE}


def test_build_leaves_are_what_reaches_the_coil(full):
    """The handoff's finding: among the sampled rows, the coil group is reached by
    exactly `f_j_tf_wp_critical_max` and `dx_tf_wp_insulation`.
    """
    stages = split(full, rows(full)["sampled"])
    hits, _ = violations(full, stages)
    by_place = {v.place: v for v in hits}
    assert by_place[".tfcoil.j_tf_wp"].responsible == (
        ".constraints.f_j_tf_wp_critical_max",
        ".tfcoil.dx_tf_wp_insulation",
    )
    assert by_place[".stellarator.wp_width_r_min"].responsible == (
        ".constraints.f_j_tf_wp_critical_max",
    )


def test_no_block_straddles_the_boundary(full):
    """A component lands whole in one stage, and the nesting survives the split."""
    stages = split(full, rows(full)["build"])
    for component in full.graph.components:
        assert len({stages.stage[n] for n in component}) == 1
    first = first_stage_graph(full, stages)
    recourse = recourse_graph(full, stages)
    assert set(first.nodes) | set(recourse.nodes) == set(full.nodes)
    assert not set(first.nodes) & set(recourse.nodes)
    assert len(first.within) + len(recourse.within) == len(full.within)
    # The first stage reads no varying leaf; the recourse reads the first stage.
    assert not set(first.graph.boundary_inputs) & set(stages.leaves.varying)
    crossing = hoisted_inputs(full, stages)
    assert crossing
    assert set(crossing) <= set(first.graph.owned_variables)
    assert set(crossing) <= set(recourse.graph.boundary_inputs)


def _same(a, b, rtol=1e-12):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.shape != b.shape:
        return False
    both_nan = np.isnan(a) & np.isnan(b)
    scale = np.maximum(np.abs(a), np.abs(b))
    return bool(np.all(both_nan | (np.abs(a - b) <= rtol * scale)))


def test_hoist_reproduces_the_whole_graph(configuration):
    """First stage once, then the recourse over it, equals the whole MDA at the
    nominal on every shared place.
    """
    reference = native.reference_of(configuration)
    _driven, runnable, schedule, _run = mda_schedule(graph_for(configuration.machine))
    env = seed_env(reference.cold, schedule, runnable)
    whole = run_schedule(schedule, env)

    stages = split(runnable, rows(runnable)["build"])
    split_run = hoisted(runnable, stages, env)
    out = split_run()
    joined = {**split_run.first_env, **out}

    assert set(whole) <= set(joined)
    # Enough of it: a first stage worth hoisting, and the recourse reads it.
    first_inputs = set(split_run.first.inputs)
    assert first_inputs < set(env)
    assert len(split_run.first_env) - len(first_inputs) >= len(stages.first)
    assert set(hoisted_inputs(runnable, stages)) <= set(split_run.recourse.inputs)
    differing = [var.spelling for var in whole if not _same(whole[var], joined[var])]
    assert not differing, differing[:20]

    # An override at a varying leaf moves the recourse and not the first stage; one
    # at a first-stage place is refused.
    density = next(v for v in stages.leaves.operating if v.spelling == RECOURSE_PATHS[0])
    moved = split_run({density: env[density] * 1.01})
    assert not _same(moved[density], whole[density])
    assert set(moved) == set(out)
    hoisted_place = hoisted_inputs(runnable, stages)[0]
    with pytest.raises(ValueError, match="first stage"):
        split_run({hoisted_place: joined[hoisted_place]})
