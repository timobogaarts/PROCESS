"""`mda.py` builds a runnable `Schedule` for the whole registered graph.

Not a numerical test (nothing here checks a value against PROCESS -- that's the
block-by-block comparison harness's job, a separate, larger piece) -- this only pins
the structural claim: every one of `indat.GRAPH`'s 11 SCCs, after the two
`FixedPointCut`s, declares exactly one problem, and every driven block gets a driver.
Both facts fail loudly (a `Schedule`/`Drive`/`Blocking` construction error) if either
stops being true -- see `Schedule`'s own docstring: "a `Schedule` that exists is
runnable."
"""

import equinox as eqx
from cottax.blocking import Blocking
from cottax.evaluate import schedule_for
from cottax.interfaces.pytree_namespace_module import to_graph
from cottax.problem import Driven, FixedPoint, RootFind, Start, driver_vars
from cottax.rewrites import Cut

from functional_process.boundary import TOKAMAK_INPUT_FILE
from functional_process.core.solver.drivers import PicardDriver, SeededNewtonDriver
from functional_process.indat import (
    GRAPH,
    REFERENCE_MACHINE,
    WINDING_PACK_MATERIAL,
    graph_for,
    machine_from_indat,
)
from functional_process.mda import (
    CUTS,
    ROOT_FIND_SEEDS,
    default_drivers,
    driven_graph,
    schedule,
    starts_for,
)
from process.models.superconductors import SuperconductorModel


def _cut_all(sub, vars_):
    """`sub` with every variable in `vars_` cut, or `None` if one of them has no
    closing reader left by the time its turn comes (i.e. an earlier cut already broke
    every path it closed -- which is what makes it redundant).
    """
    out = sub
    for v in vars_:
        readers = out.closing_readers(v)
        if not readers:
            return None
        out = Cut(var=v, readers=readers).apply(out)
    return out


def _assert_every_raw_cycle_is_cut_sufficiently_and_minimally(graph, machine):
    """`graph`'s `CUTS` entries break every one of its raw cycles completely, and
    dropping any one of them leaves that cycle cyclic -- sufficient *and* minimal.

    Takes the graph rather than reading `GRAPH`, because `CUTS` is now one table serving
    two machines and the property has to hold on each separately: an entry can be
    necessary on one and absent from the other's graph entirely (the tokamak's
    `dx_tf_wp_primary_toroidal`, the stellarator's -- nothing, as it happens), and a
    check that only ever ran on the stellarator would not have noticed that the pedestal
    profile arm needs a third density cut.
    """
    for cycle in graph.cycles:
        names = {n.path_str() for n in cycle}
        if any(n.startswith("^problem") for n in names):
            continue  # already a declared block, not a raw cycle to cut
        sub = graph.subgraph(cycle)
        cutting_vars = [v for v in CUTS if v in sub.owners]
        assert cutting_vars, (
            f"[{machine}] cycle {sorted(names)} has no CUTS entry among its owned "
            f"variables"
        )
        assert _cut_all(sub, cutting_vars).is_acyclic, (
            f"[{machine}] cycle {sorted(names)} is still cyclic after cutting "
            f"{[v.path_str() for v in cutting_vars]}"
        )
        for dropped in cutting_vars:
            rest = [v for v in cutting_vars if v != dropped]
            without = _cut_all(sub, rest)
            assert without is None or not without.is_acyclic, (
                f"[{machine}] {dropped.path_str()} is redundant: cycle {sorted(names)} "
                f"is already acyclic without it, so it should not be in CUTS"
            )


def test_each_raw_cycle_is_fully_broken_by_its_own_cuts_and_no_fewer():
    """The property above, on the reference (stellarator) machine.

    Pinned so a future edit to a cycle's membership (a new node reading or owning
    something in it) is forced to re-check rather than silently keep a now-partial cut.
    That is not hypothetical: registering `FusionTotalsNoBeam` added a second
    `FusionRates -> PlasmaComposition` path and made the single
    `proton_rate_density` cut insufficient, which `Blocking` caught only because it
    refuses a block that is *"still cyclic with its problem(s) removed"*. This test
    now states the property directly. It re-derives `mda.CUTS`'s own claim rather than
    trusting the docstring.
    """
    _assert_every_raw_cycle_is_cut_sufficiently_and_minimally(GRAPH, "stellarator")


def test_each_raw_cycle_is_fully_broken_on_the_tokamak_too():
    """The same property on `large_tokamak_eval.IN.DAT` -- four raw cycles, eight cuts
    between them, none of them redundant.

    Worth its own test rather than a second loop inside the first, because the two
    machines fail differently and the message should say which. The tokamak's density
    cycle is 8 nodes to the stellarator's 6 (`i_plasma_pedestal = 1` puts
    `pedestal_profile_values`/`ne_profile_integral` in the profile slot) and needs the
    third cut `mda.CUTS` documents; its build/winding-pack, volt-second/burn-time and
    PF coil cycles have no stellarator counterpart at all.
    """
    _assert_every_raw_cycle_is_cut_sufficiently_and_minimally(
        graph_for(machine_from_indat(TOKAMAK_INPUT_FILE)), "tokamak"
    )


def test_the_tokamak_build_winding_pack_cycle_is_cut_where_process_reads_stale():
    """The 4-node build/winding-pack cycle on `large_tokamak_eval.IN.DAT`: every one of
    its four edge variables is a *sufficient* single cut, `CUTS` names exactly one of
    them, and the one it names is the edge PROCESS itself reads stale.

    Same shape as the density loop's test above, plus the tie-break that loop did not
    need. There, one candidate out of 42 worked and measurement settled it outright;
    here **all four candidates work structurally**, so a test that only asserted
    sufficiency would pass for any of the four and pin nothing. What it pins instead is
    the semantic choice: of the four edges, three are read fresh within one
    `Caller._call_models_once` pass and one crosses the pass boundary, and `CUTS` names
    that one. See `mda.CUTS`'s own docstring for the `caller.py`/`build.py` line
    references behind "reads stale".

    Minimality is the empty set: a single cut is minimal iff dropping it leaves the
    cycle cyclic, which is asserted directly.
    """
    graph = graph_for(machine_from_indat(TOKAMAK_INPUT_FILE))
    (cycle,) = [
        c
        for c in graph.cycles
        if {n.path_str() for n in c}
        == {
            ".tokamak.build.tf_outboard_mid",
            ".tokamak.build.wp_conductor_max_width",
            ".tokamak.cicc_superconducting_tf_coil.superconducting_tf_wp_geometry",
            ".tokamak.cicc_superconducting_tf_coil.tf_global_geometry",
        }
    ]
    sub = graph.subgraph(cycle)

    # Sufficiency, candidate by candidate -- the measurement, re-derived rather than
    # quoted. `closing_readers` is empty for the 18 owned variables that leave the cycle
    # without re-entering it, so the candidate set is exactly the four edges.
    sufficient = set()
    for var in sub.owners:
        readers = sub.closing_readers(var)
        if not readers:
            continue
        if Cut(var=var, readers=readers).apply(sub).is_acyclic:
            sufficient.add(var.path_str())
    assert sufficient == {
        ".superconducting_tfcoil.tan_theta_coil",
        ".tfcoil.dx_tf_wp_primary_toroidal",
        ".tfcoil.dx_tf_wp_conductor_max",
        ".build.r_tf_outboard_mid",
    }

    # ... and the one `CUTS` picks out of those four is the stale-read edge.
    chosen = [v.path_str() for v in CUTS if v in sub.owners]
    assert chosen == [".tfcoil.dx_tf_wp_primary_toroidal"]
    # Minimal, i.e. not vacuous: without it the cycle is still a cycle.
    assert not sub.is_acyclic

    # And the whole tokamak graph is runnable with it -- the property `Blocking` refused
    # ("coupled block declares no problem") before this cut existed.
    blocking = Blocking.scc(driven_graph(graph))
    for block, problem_type in zip(blocking.blocks, blocking.problem_types, strict=True):
        assert len(block) == 1 or problem_type is not None, block


def test_the_tokamak_density_cycle_is_cut_at_the_variable_process_bootstraps():
    """The pedestal arm's extra ring: four candidates finish the cut, `CUTS` names the
    one PROCESS itself carries across a pass.

    Same shape as the build/winding-pack test above and the same reason for existing --
    sufficiency does not pick. With `proton_rate_density` and `fusden_alpha_total`
    already cut, exactly four single variables make the remaining 5-node ring acyclic,
    and the one chosen is the one `physics.py:1377-1387` reads from the *previous* call
    of `plasma_profile.run()` behind a `first_call` bootstrap.
    """
    graph = graph_for(machine_from_indat(TOKAMAK_INPUT_FILE))
    (cycle,) = [
        c
        for c in graph.cycles
        if any(n.path_str() == ".physics.profiles.density_profile" for n in c)
        and not any(n.path_str().startswith("^problem") for n in c)
    ]
    sub = graph.subgraph(cycle)
    shared = {".physics.proton_rate_density", ".physics.fusden_alpha_total"}
    first_two = [v for v in CUTS if v.path_str() in shared]
    partial = _cut_all(sub, first_two)
    assert not partial.is_acyclic, (
        "the stellarator's two density cuts already break the tokamak's cycle -- the "
        "third cut would be redundant and belongs out of CUTS"
    )
    finishes = {
        v.path_str()
        for v in sub.owners
        if v not in first_two and (g := _cut_all(sub, [*first_two, v])) and g.is_acyclic
    }
    assert finishes == {
        ".physics.f_temp_plasma_electron_density_vol_avg",
        ".physics.nd_plasma_electron_on_axis",
        ".physics.nd_plasma_electron_profile",
        ".physics.nd_plasma_ions_total_vol_avg",
    }
    chosen = [v.path_str() for v in CUTS if v in sub.owners]
    assert chosen[2:] == [".physics.f_temp_plasma_electron_density_vol_avg"]


def test_the_volt_second_burn_time_cycle_is_cut_where_process_reads_stale():
    """The two-node volt-second/burn-time ring on `large_tokamak_eval.IN.DAT`: both
    edge variables are sufficient single cuts, `CUTS` names exactly one, and the one it
    names is the value PROCESS itself carries across a pass.

    Same shape as the build/winding-pack test. `physics.run()` (volt-seconds inside)
    runs before `pulse.run()` in one `Caller._call_models_once` pass, so
    `.physics.v_plasma_loop_burn` is read fresh and `.times.t_plant_pulse_burn` is the
    previous pass's -- PROCESS's own comment at `physics.py:4882-4884` says so
    outright. See `mda.CUTS`'s docstring.
    """
    graph = graph_for(machine_from_indat(TOKAMAK_INPUT_FILE))
    (cycle,) = [
        c
        for c in graph.cycles
        if {n.path_str() for n in c}
        == {".tokamak.plasma_inductance.volt_seconds", ".tokamak.pulse.burn_time"}
    ]
    sub = graph.subgraph(cycle)
    sufficient = {
        v.path_str()
        for v in sub.owners
        if (readers := sub.closing_readers(v))
        and Cut(var=v, readers=readers).apply(sub).is_acyclic
    }
    assert sufficient == {".physics.v_plasma_loop_burn", ".times.t_plant_pulse_burn"}
    chosen = [v.path_str() for v in CUTS if v in sub.owners]
    assert chosen == [".times.t_plant_pulse_burn"]
    assert not sub.is_acyclic


def test_the_pf_coil_cycle_is_cut_at_the_variables_process_seeds():
    """The five-node PF coil ring on `large_tokamak_eval.IN.DAT`: sufficiency alone
    allows exactly two single cuts, neither of which is what PROCESS carries, and
    `CUTS` names instead the **pair PROCESS itself seeds** -- `pfcoil.py:605-608`'s
    `first_call` bootstrap writes `ind_pf_cs_plasma_mutual[:, :] = 1.0` and
    `n_pf_coil_turns[:] = 100.0`, so those two are the iteration's loop-carried
    unknowns by PROCESS's own declaration. Each is necessary given the other
    (dropping either leaves the ring cyclic), which is the minimality the shared
    checker above also re-asserts for every cycle.

    The measurement, re-derived rather than quoted: eleven of the ring's fifteen owned
    variables have closing readers; `.pf_coil.c_pf_cs_coils_peak_ma` and
    `.pf_coil.f_j_cs_start_end_flat_top` are the only sufficient single cuts, and
    each is one stale edge of the three PROCESS actually carries across a pass, so
    neither reproduces PROCESS's recurrence -- see `mda.CUTS`'s docstring for the
    tie-break, and the round-2 brief for the FixedPointCut -> Picard decision
    (`RootFind` on the `n_pf_coil_turns` residual is the recorded later upgrade).
    """
    graph = graph_for(machine_from_indat(TOKAMAK_INPUT_FILE))
    (cycle,) = [
        c
        for c in graph.cycles
        if {n.path_str() for n in c}
        == {
            ".tokamak.cs_coil.flux_swing",
            ".tokamak.pf_coil.inductance",
            ".tokamak.pf_coil.sizes",
            ".tokamak.pf_coil.time_point_currents",
            ".tokamak.pf_coil.waveform",
        }
    ]
    sub = graph.subgraph(cycle)

    candidates = [v for v in sub.owners if sub.closing_readers(v)]
    assert len(candidates) == 11
    sufficient = {
        v.path_str()
        for v in candidates
        if Cut(var=v, readers=sub.closing_readers(v)).apply(sub).is_acyclic
    }
    assert sufficient == {
        ".pf_coil.c_pf_cs_coils_peak_ma",
        ".pf_coil.f_j_cs_start_end_flat_top",
    }

    chosen = [v for v in CUTS if v in sub.owners]
    assert [v.path_str() for v in chosen] == [
        ".pf_coil.ind_pf_cs_plasma_mutual",
        ".pf_coil.n_pf_coil_turns",
    ]
    # Sufficient together, and neither redundant -- the pair is PROCESS's seed set.
    assert _cut_all(sub, chosen).is_acyclic
    for dropped in chosen:
        rest = [v for v in chosen if v != dropped]
        without = _cut_all(sub, rest)
        assert without is None or not without.is_acyclic, dropped.path_str()

    # And the whole tokamak graph is runnable with the cuts -- every cyclic block
    # declares a problem and carries a driver.
    blocking = Blocking.scc(driven_graph(graph))
    for block, problem_type in zip(blocking.blocks, blocking.problem_types, strict=True):
        assert len(block) == 1 or problem_type is not None, block


def test_the_tokamak_only_cuts_leave_the_stellarator_graph_untouched():
    """`CUTS`'s two tokamak-only entries are inert on the reference (stellarator)
    machine.

    `cut_graph` only cuts a variable that actually has closing readers in the graph it
    was handed, and on the stellarator neither has any: the winding-pack cycle's nodes do
    not exist (its TF coils are `models/stellarator/coils/`), and
    `.physics.f_temp_plasma_electron_density_vol_avg` is owned by the *parabolic* profile
    occupant, which nothing in the cycle reads back into. Stated as a test rather than
    trusted, because the failure mode is silent: a cut that *did* land would mint an
    unknown, open a `^guess.*` boundary input and move every pinned boundary count, and
    the first thing to notice would be an unrelated harness number.
    """
    graph = driven_graph()
    for name in (
        "dx_tf_wp_primary_toroidal",
        "f_temp_plasma_electron_density_vol_avg",
        "t_plant_pulse_burn",
        "ind_pf_cs_plasma_mutual",
        "n_pf_coil_turns",
    ):
        # A landed cut mints a `^guess.*` port and declares a `^problem.*`; a *plain*
        # boundary input spelled with the same name is fine -- `.times.
        # t_plant_pulse_burn` legitimately IS one on the stellarator (`PulseDurations`
        # reads it, nothing produces it, and the pin records it as `input`).
        assert not any(
            name in v.path_str() and v.path_str().startswith("^guess")
            for v in graph.unowned_inputs
        ), name
        assert not any(name in p.path_str() for p in graph.declared), name


def test_driven_graph_has_no_raw_cycles_left():
    """After both cuts, every genuinely *cyclic* block (more than one node) declares
    exactly one problem -- `Blocking.scc` would raise `Graph.problem_type`'s refusal
    otherwise. Singleton, acyclic blocks are expected to have `problem_type is None`
    ("run, not driven") -- most of the graph's 98 nodes are ordinary acyclic ones.
    """
    graph = driven_graph()
    blocking = Blocking.scc(graph)
    for block, problem_type in zip(blocking.blocks, blocking.problem_types, strict=True):
        if len(block) > 1:
            assert problem_type is not None, (
                f"cyclic block {block!r} declares no problem"
            )


def test_default_drivers_assigns_newton_to_root_find_and_picard_to_fixed_point():
    """Every driven block gets the driver matching its own declared problem type --
    still assigned by type, not per block. The `RootFind` driver is
    `SeededNewtonDriver` (a Newton with a fallback starting guess, see its docstring);
    which *guess* it falls back to is per-unknown, but the driver choice is not.
    """
    graph = driven_graph()
    blocking = Blocking.scc(graph)
    # Read off the graph, not from `default_drivers`: `driven_graph` has already
    # `Assign`ed every driver, and `default_drivers` skips a problem that carries one --
    # so asking it again would return an empty map. The driver is a property of the node
    # now, which is the whole point of the change.
    drivers = {
        name: node.driver
        for name, node in graph.definitions.items()
        if isinstance(node, Driven)
    }

    for problem, problem_type in zip(
        blocking.problems, blocking.problem_types, strict=True
    ):
        if problem_type is None:
            continue
        driver = drivers[problem]
        if issubclass(problem_type, RootFind):
            assert isinstance(driver, SeededNewtonDriver)
        elif issubclass(problem_type, FixedPoint):
            assert isinstance(driver, PicardDriver)


def test_every_root_find_unknown_has_a_starting_guess_that_does_not_need_data():
    """Every `RootFind` block in the graph can name a starting guess without `data` --
    either supplied by a node (`SUPPLIED_STARTS`) or as a driver-side fallback
    (`ROOT_FIND_SEEDS`).

    Pinned because the failure it prevents is not local: seeded from a cold
    `DataStructure`, `Intersect`'s unknown is `0.0`, the residual is exactly flat there,
    and `optimistix` aborts -- taking the **whole schedule** down, not just its own
    block. A new `RootFind` with neither kind of guess would reintroduce that the moment
    anyone ran the port cold.

    **Two mechanisms, and the supplied one is the better half.** `Intersect`'s guess is
    computed by PROCESS itself, so the occupant of `winding_pack_intersect_inputs` owns
    it and `Supply` points the `Start` port at it (`_audit/next_steps.md` §14.5) -- no
    `data`, no block context, no fallback. `d_duct`'s cannot be: PROCESS writes a
    literal and nothing computes it. Accepting either is what makes this test the
    question it means to ask ("can this block start cold?") rather than a check that one
    particular table has a row.
    """
    graph = driven_graph()
    blocking = Blocking.scc(graph)
    for problem, problem_type in zip(
        blocking.problems, blocking.problem_types, strict=True
    ):
        if problem_type is None or not issubclass(problem_type, RootFind):
            continue
        unknowns = graph[problem].owns
        # A `Supply`-ed start is a `Start` port the graph owns -- `starts_for` filters
        # exactly those out, since there is nothing left for a caller to seed.
        supplied = {u for u, _ in starts_for(graph, problem)} != set(unknowns)
        assert supplied or any(u.path_str() in ROOT_FIND_SEEDS for u in unknowns), (
            f"{problem.path_str()} solves for "
            f"{[u.path_str() for u in unknowns]}: no `Start` is supplied by a node and "
            f"none has a `ROOT_FIND_SEEDS` entry -- it would fail from a cold start"
        )


def test_the_intersect_start_is_supplied_by_the_winding_pack_occupant():
    """`Supply`, in place: `^guess.stellarator.wp_width_r_min` is not a boundary input of
    the driven graph -- the `Start` port reads `.stellarator.wp_width_r_min_guess`, which
    the `winding_pack_intersect_inputs` occupant owns.

    The sharp end of §14.5. `ROOT_FIND_SEEDS` used to derive this guess from
    `.stellarator.r_coil_minor` read out of the block's *context*, and `r_coil_minor` was
    only in that context because a switch kwarg made the pre-`intersect` node declare
    `.tfcoil.j_tf_wp` on every material -- the invented edge that closed the block. With
    the occupant split there is no such edge and no such context; with `Supply` there
    does not need to be one.
    """
    graph = driven_graph()
    (problem,) = [
        name
        for name in graph.declared
        if name.path_str() == "^problem.stellarator.coils.intersect"
    ]
    starts = driver_vars(graph[problem], Start)
    assert [s.path_str() for s in starts] == [".stellarator.wp_width_r_min_guess"]
    assert starts[0] in graph.owners
    assert not any(
        v.path_str().startswith("^guess.stellarator.wp_width_r_min")
        for v in graph.unowned_inputs
    )


def test_every_superconductor_schedules_and_only_bi2212_keeps_its_guess():
    """`supply_starts` is conditional, and this is the condition.

    cottax refuses a `Start` produced inside the block it starts. `Bi2212...` is the one
    occupant of `winding_pack_intersect_inputs` that reads `.tfcoil.j_tf_wp`, so with it
    the guess's producer is *in* the coils SCC and the supply must be skipped -- leaving
    `^guess.stellarator.wp_width_r_min` a boundary input, which is the honest answer for
    a guess that is not available until the solve computing it has run. With the other
    seven the supply lands and the port leaves the boundary.

    Every one of the eight builds a `Schedule`, which is the check that the skip is a
    skip and not a latent `schedule_for` refusal waiting for someone to select that
    material (`_audit/next_steps.md` §14.5).
    """
    for material, occupant in WINDING_PACK_MATERIAL.items():
        machine = eqx.tree_at(
            lambda m: m.stellarator.coils.winding_pack_intersect_inputs,
            REFERENCE_MACHINE,
            occupant(),
        )
        graph = driven_graph(to_graph(machine))
        schedule_for(Blocking.scc(graph))  # raises if the block cannot be driven
        at_boundary = [
            v
            for v in graph.unowned_inputs
            if v.path_str() == "^guess.stellarator.wp_width_r_min"
        ]
        assert bool(at_boundary) == (material is SuperconductorModel.BI2212), material


def test_schedule_builds_for_the_whole_graph():
    """The actual point of this module: one `Schedule` answering every block in
    `indat.GRAPH`, not just a hand-picked slice.
    """
    s = schedule()
    assert len(s.blocking.blocks) > 0
