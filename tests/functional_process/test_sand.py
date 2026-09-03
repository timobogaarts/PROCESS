"""Assembly-level tests for `functional_process.sand` and `VmconDriver`.

Deliberately **not** a value comparison against PROCESS: that is `sand_harness.py`'s
ladder, which needs a live 95-second PROCESS run and belongs in `run_sand_harness.py`.
What is checked here is everything that can be checked without one -- that the problem
assembled is the problem PROCESS states (which constraints, which are equalities, which
variables are the unknowns), that the driver's contract holds on a problem with a known
answer, and that the two places this layer can silently be wrong (a dropped constraint,
a mislabelled switch argument) fail loudly instead.
"""

import inspect
import re
import subprocess  # noqa: S404
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from cottax.blocking import Blocking
from cottax.evaluate import Drive, Schedule
from cottax.graph import Graph
from cottax.problem import (
    Converged,
    FixedPoint,
    Optimise,
    Start,
    Steps,
    driver_vars,
)
from cottax.rewrites import Assign
from cottax.spec import CallableNode, In, NodePath, Out, VarPath
from cottax.tools.path import path_map
from jax.flatten_util import ravel_pytree
from jax.tree_util import GetAttrKey, SequenceKey

import functional_process
from functional_process.core.solver import constraints as ported_constraints
from functional_process.core.solver import objectives as ported_objectives
from functional_process.core.solver.drivers import (
    VMCON_CONVERGED,
    VMCON_NON_FINITE,
    VMCON_STATUS,
    NonFiniteProblemError,
    SlsqpDriver,
    Status,
    VmconDriver,
    _refuse_non_finite,
    non_finite_summary,
)
from functional_process.indat import GRAPH
from functional_process.sand import (
    NON_INPUT_FIELDS,
    REFERENCE_SWITCH_VALUES,
    SWITCH_PARAMETER_NAMES,
    _Resolver,
    constraint_nodes,
    design_bounds,
    iteration_variable_path,
    objective_node,
    optimise_graph,
    sand_graph,
    sand_schedule,
    sand_shape,
)
from functional_process.sand_harness import ground_truth, run_schedule, schedule_verdict
from functional_process.vocabulary import AREAS
from functional_process.vocabulary.input_variables import INPUT_VARIABLES
from process.core.input import INPUT_VARIABLES as PROCESS_INPUT_VARIABLES
from process.core.model import DataStructure
from process.core.solver.iteration_variables import ITERATION_VARIABLES

REPO_ROOT = Path(functional_process.__file__).resolve().parent.parent
"""Anchored on the package, not on this file, so it survives the tests living under
`tests/` while `functional_process/` stays where it is."""

REFERENCE_IXC = [2, 3, 4, 6, 10, 56, 59, 109]
REFERENCE_ICC = [2, 16, 24, 8, 17, 18, 67, 82, 83, 62, 32, 34, 35, 65]
REFERENCE_N_EQUALITY = 2
REFERENCE_FIGURE_OF_MERIT = 6
"""`stellarator_helias.IN.DAT`'s own problem. Transcribed rather than read off a live
run so these tests need no PROCESS execution; `run_sand_harness.py` reads the real
`numerics` and would disagree loudly if the file ever changed."""


def test_reference_problem_matches_the_input_file():
    """The four constants above really are what the reference `IN.DAT` asks for.

    Same discipline as `test_machine.py::
    test_reference_configuration_matches_the_input_file`: a transcription that nothing
    checks is a transcription that drifts. `ixc`/`icc` are not parsed (they are
    multi-line lists with per-entry comments); the three scalars that decide the
    *shape* of the problem are.
    """
    text = (
        REPO_ROOT / "tests/regression/input_files/stellarator_helias.IN.DAT"
    ).read_text()
    settings = {
        m.group(1): int(m.group(2))
        for line in text.splitlines()
        if (m := re.match(r"\s*([A-Za-z_]\w*)\s*=\s*(-?\d+)\s*(\*.*)?$", line))
    }
    assert settings["n_equality_constraints"] == REFERENCE_N_EQUALITY
    assert settings["i_figure_merit"] == REFERENCE_FIGURE_OF_MERIT
    # `n_inequality_constraints` is *derived*, not set: `init.set_active_constraints`
    # takes it as `len(icc) - n_equality_constraints`
    # (`process/core/init.py:1277-1294`), which is why the split is positional in the
    # first place.
    assert len(REFERENCE_ICC) > REFERENCE_N_EQUALITY


def _contract_static_argnames():
    """`{constraint id: static_argnames}` read out of `test_constraints.py`'s own
    `Tier1Contract` classes -- the maintained source of truth for which parameter of a
    ported constraint is a switch.
    """
    source = (
        Path(__file__).resolve().parent / "core/solver/test_constraints.py"
    ).read_text()
    out = {}
    for block in source.split("\nclass "):
        ported = re.search(r"ported = constraint_(\d+)", block)
        if not ported:
            continue
        declared = re.search(r"static_argnames = \(([^)]*)\)", block)
        names = (
            tuple(
                part.strip().strip("\"'")
                for part in declared.group(1).split(",")
                if part.strip()
            )
            if declared
            else ()
        )
        out[int(ported.group(1))] = names
    return out


def test_switch_values_match_the_contracts_static_argnames():
    """`REFERENCE_SWITCH_VALUES` decides, mechanically, which parameters are frozen at
    assembly time. Both directions matter and both are checked.

    A switch that leaked onto a port would be traced into an `if` (a `TracerBoolConversion`
    at best, a silently-wrong branch at worst); a real variable frozen into the partial
    would be a constant where a derivative belongs, which is exactly the defect class
    Stage B exists to find. Cross-checking against the `Tier1Contract`s means the two
    declarations cannot drift apart silently.
    """
    contracts = _contract_static_argnames()
    for cid in REFERENCE_ICC:
        fn = getattr(ported_constraints, f"constraint_{cid}")
        parameters = set(inspect.signature(fn).parameters)
        assert parameters & set(REFERENCE_SWITCH_VALUES) == set(contracts[cid]), (
            f"constraint {cid}: sand would bind "
            f"{sorted(parameters & set(REFERENCE_SWITCH_VALUES))} statically, but its "
            f"Tier1Contract declares static_argnames={contracts[cid]}"
        )


def test_iteration_variable_path_is_processs_own_accessor():
    """`ITERATION_VARIABLES[id]`'s `(module, target_name or name)` is the `VarPath`."""
    for i in REFERENCE_IXC:
        iteration_variable = ITERATION_VARIABLES[i]
        assert iteration_variable_path(i) == VarPath((
            GetAttrKey(iteration_variable.module),
            GetAttrKey(iteration_variable.target_name or iteration_variable.name),
        ))


def test_array_indexed_iteration_variables_get_a_sequence_key():
    """IDs 125-136 address one element of `f_nd_impurity_electron_array`.

    This test asserted the opposite until 2026-08-30 -- that such a variable is
    *refused* -- because `Graph.__check_init__` rejects an output lying inside a
    variable another node reads whole, and `_audit/optimise_design.md` §1.3 recorded
    that every consumer of impurity fractions read the whole array.

    The premise expired without the refusal noticing. `radiation_power.py` now reads the
    array as fourteen individual index-addressed ports, so nothing reads the enclosing
    array and the containment rule has nothing to object to. The cost of the stale
    refusal was three reference configurations -- `large_tokamak_nof`,
    `low_aspect_ratio_DEMO` and `st_regression` -- that could not reach a harness stage,
    two of which assemble in both formulations the moment it is lifted.

    The lesson is the reason this docstring is long: a refusal is a claim about the code
    at the time it was written, and nothing re-checks it. This one is now a positive
    assertion, which fails if the path stops being addressable rather than silently
    passing forever.
    """
    for identifier in (125, 126, 135):
        iteration_variable = ITERATION_VARIABLES[identifier]
        assert iteration_variable.array_index is not None
        assert iteration_variable_path(identifier) == VarPath((
            GetAttrKey(iteration_variable.module),
            GetAttrKey(iteration_variable.target_name or iteration_variable.name),
            SequenceKey(iteration_variable.array_index),
        ))


def test_the_whole_impurity_array_is_read_by_nobody():
    """The premise the test above depends on, pinned separately so it cannot expire in
    silence the way the refusal did.

    An `Optimise` may own `f_nd_impurity_electron_array[12]` only while no node reads
    the enclosing array. If a future model reads it whole, `Graph.__check_init__` will
    refuse the *graph*, which is a confusing place to learn this; failing here says why.
    """
    from functional_process.indat import graph_for

    graph = graph_for()
    array = VarPath((
        GetAttrKey("impurity_radiation"),
        GetAttrKey("f_nd_impurity_electron_array"),
    ))
    readers = [
        node.path_str()
        for node in graph.nodes
        if any(read == array for read in graph[node].reads)
    ]
    assert not readers, (
        f"{readers} read `.impurity_radiation.f_nd_impurity_electron_array` whole, so "
        f"an `Optimise` can no longer own one of its elements -- see "
        f"`sand.iteration_variable_path`"
    )


def test_all_eight_design_variables_are_boundary_inputs():
    """An `Optimise` owning them therefore collides with nothing.

    This is what makes the whole layer cheap: registering the problem turns eight free
    inputs into eight owned variables and changes nothing else. Seven of the 83-entry
    table *are* produced by a node (ID 1 `.physics.aspect` among them) and would be a
    duplicate-ownership conflict; none of those is active in this run.
    """
    owned = set(GRAPH.owners)
    for i in REFERENCE_IXC:
        assert iteration_variable_path(i) not in owned


def test_a_field_process_never_writes_seeds_as_nan_not_as_zero():
    """`.tfcoil.sig_tf_cs_bucked` is `None` in PROCESS's own converged `DataStructure`
    on `large_tokamak_eval` -- `stresscl` assigns it only at `i_tf_bucking >= 2`
    (`process/models/tfcoil/base.py:3235`) -- and `jnp.asarray(None)` is what stopped
    the tokamak SAND harness while building a `Drive`'s context.

    The seed is `nan` **because the read is claimed to be dead**, and a claim that
    cheap should be checked on every run rather than trusted: a `0.0` placeholder would
    let a read that is not actually dead produce a plausible number, where `nan`
    propagates into the condition and the pre-solve probe already stops on a non-finite
    one. `KNOWN_MINT_VALUES` still wins, so this cannot shadow a mint's analytic value.
    """

    class _Area:
        sig_tf_cs_bucked = None
        stress_shear_cs_peak = 1.1647e9

    class _Data:
        tfcoil = _Area()
        pf_coil = _Area()

    unwritten = VarPath((GetAttrKey("tfcoil"), GetAttrKey("sig_tf_cs_bucked")))
    written = VarPath((GetAttrKey("pf_coil"), GetAttrKey("stress_shear_cs_peak")))
    assert np.isnan(ground_truth(_Data(), unwritten))
    assert ground_truth(_Data(), written) == 1.1647e9


def test_constraint_72s_free_standing_arm_is_unmoved_by_the_nan_seed():
    """The other half of the seed's claim: on `i_tf_bucking = 1` the value `nan` seeds
    reaches no arithmetic, so the condition is identical to what any other placeholder
    would give.

    This is the *deadness* itself, pinned. `sand._bind` declares an `In` for every
    non-switch parameter of a constraint whether or not the statically selected arm
    consumes it, so c72 carries a read of `sig_tf_cs_bucked` that only the
    bucked-and-wedged arm (`i_tf_bucking >= 2` **and** `i_tf_inside_cs ==
    TF_OUTSIDE_CS`) can reach -- a dead read, not a missing producer
    (`_audit/units/models/tfcoil/superconducting.md`). The bucked arm is checked too,
    so a future machine that takes it cannot inherit the `nan` silently.
    """
    free_standing = dict(
        i_tf_bucking=1,
        i_tf_inside_cs=0,
        stress_shear_cs_peak=1.1647e9,
        stress_cs_steel_max=6.6e8,
    )
    with_nan = ported_constraints.constraint_72(
        sig_tf_cs_bucked=float("nan"), **free_standing
    )
    with_number = ported_constraints.constraint_72(sig_tf_cs_bucked=0.0, **free_standing)
    assert with_nan == with_number
    assert np.isfinite(with_nan[1])

    # And the other arm, which is the one that reads it. This assertion is the reverse
    # of what it said until 2026-08-30: the bucked-and-wedged arm used Python's builtin
    # `max`, and because `nan > x` is `False` the builtin returns `x` -- so the sentinel
    # was silently discarded and this test pinned that as a measured limitation of the
    # guard. It was not only a limitation. The builtin also calls `bool()` on `b > a`,
    # which raises `TracerBoolConversionError` under `jit`, and that single line made
    # the whole tokamak MDF problem untraceable (`_audit/optimise_design.md` §16). The
    # fix -- `jnp.maximum` -- is required for tracing and propagates the `nan` as a
    # side effect, so the guard now works on both arms and this pins the alarm instead
    # of the hole in it.
    bucked = ported_constraints.constraint_72(
        **{**free_standing, "i_tf_bucking": 2}, sig_tf_cs_bucked=float("nan")
    )
    assert not np.isfinite(bucked[1]), (
        "the bucked arm reads `sig_tf_cs_bucked`, so an unwritten value must reach the "
        "residual rather than being swallowed by a comparison against `nan`"
    )
    # The same arm with a real number is unaffected -- the alarm is on the sentinel,
    # not on the branch.
    assert np.isfinite(
        ported_constraints.constraint_72(
            **{**free_standing, "i_tf_bucking": 2}, sig_tf_cs_bucked=0.0
        )[1]
    )


def test_constraint_16_is_an_equality_despite_its_geq_body():
    """The equality/inequality split is **positional and user-chosen**, not a property
    of the constraint function.

    `stellarator_helias.IN.DAT:12` sets `n_equality_constraints = 2`, so `icc[:2]` --
    constraints 2 and 16 -- are the equalities. Constraint 16's ported body is a `geq`
    ("net electric power lower limit") and PROCESS still drives it to equality. Anyone
    "fixing" `constraint_nodes` to read the body would silently change the problem.
    """
    nodes, equalities, inequalities, omitted = constraint_nodes(
        GRAPH, REFERENCE_ICC, REFERENCE_N_EQUALITY
    )
    assert omitted == {}
    assert len(nodes) == len(REFERENCE_ICC)
    assert [v.path_str() for v in equalities] == [
        "^cond.constraints.c2",
        "^cond.constraints.c16",
    ]
    assert len(inequalities) == len(REFERENCE_ICC) - REFERENCE_N_EQUALITY


def test_an_unassemblable_constraint_raises_rather_than_being_dropped():
    """Silently solving 13 of PROCESS's 14 constraints is a different problem.

    Constraint 76 is the standing instance: its `f_nd_impurity_electrons` argument is
    an *array element* (`_audit/optimise_design.md` §3), which name-based resolution
    cannot reach. It is asserted to be the only unassemblable one of the ~82 ported
    constraints, so this test also records that fact rather than merely using it.
    """
    unassemblable = sorted(
        cid
        for name in dir(ported_constraints)
        if name.startswith("constraint_")
        and (cid := int(name.removeprefix("constraint_")))
        and _cannot_resolve(cid)
    )
    assert unassemblable == [76]
    unresolvable = unassemblable[0]
    with pytest.raises(ValueError, match="cannot be assembled"):
        constraint_nodes(GRAPH, [*REFERENCE_ICC, unresolvable], REFERENCE_N_EQUALITY)
    _nodes, _eq, _ineq, omitted = constraint_nodes(
        GRAPH,
        [*REFERENCE_ICC, unresolvable],
        REFERENCE_N_EQUALITY,
        omit={unresolvable},
    )
    assert unresolvable in omitted


def _cannot_resolve(cid):
    try:
        constraint_nodes(GRAPH, [cid], 0)
    except ValueError:
        return True
    return False


def test_objective_node_folds_in_the_sign_and_owns_one_minted_variable():
    """`Optimise` minimises, so a negative `i_figure_merit` (PROCESS's "maximise") has
    to become a negated body -- not a flag on the driver, which would make its `drives`
    claim a lie, and not a field on `Optimise`, which has none.
    """
    _name, node, var = objective_node(GRAPH, REFERENCE_FIGURE_OF_MERIT)
    assert var.path_str() == "^cond.numerics.objf"
    assert len(node.outputs) == 1
    _name, negated, _var = objective_node(GRAPH, -REFERENCE_FIGURE_OF_MERIT)
    coe = 121.5
    assert float(node.fn(coe)) == pytest.approx(-float(negated.fn(coe)))


def test_sand_assembles_and_orders_its_conditions():
    """The whole assembly, and the ordering `VmconDriver`'s count-based split rests on.

    `Drive.conditions` is the problem node's `reads` and `Optimise.inputs` is
    `(objective, *equalities, *inequalities)`, so the objective is first and the
    equalities precede the inequalities. That is *checked* here rather than assumed,
    because it is the one thing standing between a correct solve and one that quietly
    treats an inequality as the objective.
    """
    from functional_process.mda import cut_graph
    from functional_process.mda_harness import _without_excluded

    graph = cut_graph(_without_excluded(GRAPH))
    with_problem, _name, report = optimise_graph(
        graph,
        REFERENCE_IXC,
        REFERENCE_ICC,
        REFERENCE_N_EQUALITY,
        REFERENCE_FIGURE_OF_MERIT,
    )
    combined, _residualised = sand_graph(with_problem)
    schedule = sand_schedule(combined, None)
    shape = sand_shape(schedule)
    drive = shape["drive"]

    names = [c.path_str() for c in drive.conditions]
    assert names[0] == "^cond.numerics.objf"
    # `.problem` because the node is `Driven` now: it *has* a problem rather than
    # being one, and cottax forwards only the graph-facing surface.
    definition = drive.subgraph[drive.problem].problem
    assert names[1 : 1 + len(definition.equalities)] == [
        c.var.path_str() for c in definition.equalities
    ]
    assert names[1 + len(definition.equalities) :] == [
        c.var.path_str() for c in definition.inequalities
    ]
    # The eight design variables come first among the unknowns, so the Schur reduction
    # in `sand_harness` can index them positionally.
    assert [v.path_str() for v in drive.unknowns[: len(REFERENCE_IXC)]] == [
        iteration_variable_path(i).path_str() for i in REFERENCE_IXC
    ]
    # Not the whole graph in one block: the acyclic remainder still runs as `Call` steps.
    assert shape["drive_nodes"] < len(combined.nodes)
    assert shape["schedule_steps"] > 1


def test_default_drivers_reads_the_split_off_the_problem_node():
    """`mda.default_drivers` never counts conditions -- it asks the `Optimise`."""
    from cottax.blocking import Blocking

    from functional_process.mda import cut_graph, default_drivers
    from functional_process.mda_harness import _without_excluded

    graph = cut_graph(_without_excluded(GRAPH))
    with_problem, _name, _report = optimise_graph(
        graph,
        REFERENCE_IXC,
        REFERENCE_ICC,
        REFERENCE_N_EQUALITY,
        REFERENCE_FIGURE_OF_MERIT,
    )
    combined, _residualised = sand_graph(with_problem)
    blocking = Blocking.scc(combined)
    drivers = default_drivers(blocking.graph)
    optimise = [d for d in drivers.values() if isinstance(d, VmconDriver)]
    assert len(optimise) == 1
    driver = optimise[0]
    problem = next(
        p
        for p, t in zip(blocking.problems, blocking.problem_types, strict=True)
        if t is not None and issubclass(t, Optimise)
    )
    definition = blocking.graph[problem]
    assert driver.n_equality == len(definition.equalities)
    assert driver.n_inequality == len(definition.inequalities)


def test_max_iter_reaches_the_driver_and_defaults_to_the_drivers_own():
    """A SAND block outgrew PROCESS's cap, so the caller may raise it -- and `None`
    still means *the driver's own default*, not "some number this layer picked".

    `VmconDriver.max_iter = 100` is PROCESS's `n_iteration_max` for PROCESS's own
    eight-variable problem; the stellarator's SAND block is 14 unknowns against 21
    conditions and needs 326 (`run_sand_harness.SAND_MAX_ITER`). Left un-threaded, the
    only way to say so was to edit the driver's default, which every other `Optimise`
    in the tree -- MDF's included -- would have silently inherited.
    """
    from cottax.blocking import Blocking

    from functional_process.mda import cut_graph, default_drivers
    from functional_process.mda_harness import _without_excluded

    graph = cut_graph(_without_excluded(GRAPH))
    with_problem, _name, _report = optimise_graph(
        graph,
        REFERENCE_IXC,
        REFERENCE_ICC,
        REFERENCE_N_EQUALITY,
        REFERENCE_FIGURE_OF_MERIT,
    )
    combined, _residualised = sand_graph(with_problem)
    blocking = Blocking.scc(combined)

    def only_optimise(**kwargs):
        drivers = default_drivers(blocking.graph, **kwargs)
        return next(d for d in drivers.values() if isinstance(d, VmconDriver))

    own_default = VmconDriver.__dataclass_fields__["max_iter"].default
    assert only_optimise().max_iter == own_default
    assert only_optimise(max_iter=None).max_iter == own_default
    assert only_optimise(max_iter=500).max_iter == 500

    schedule = sand_schedule(combined, None, max_iter=500)
    drive = sand_shape(schedule)["drive"]
    assert drive.driver.max_iter == 500


def test_design_bounds_are_processs_own_table():
    for var, lower, upper in design_bounds(REFERENCE_IXC):
        assert lower < upper
        assert var in {iteration_variable_path(i) for i in REFERENCE_IXC}


_EMPTY_GRAPH = Graph(path_map({}))
"""A graph with no nodes: stage one can never hit, so stage two is what is tested."""


class TestStageTwo:
    """`_Resolver`'s second stage: PROCESS's declared inputs, not a `DataStructure`.

    Stage two used to build a live `DataStructure` and ask all 36 areas which one had a
    field of the wanted name -- the port's last runtime `process` import outside the
    harnesses. It is now a lookup in the vendored `INPUT_VARIABLES` plus
    `COMPUTED_BY_AN_UNPORTED_MODEL`. These tests hold the swap to the old answer, which
    is the only thing that makes it safe: PROCESS is importable *here*, so the scan the
    port no longer performs can still be performed as an oracle.
    """

    @staticmethod
    def _scan(data, name):
        """The old stage two, verbatim: areas of `data` carrying a field `name`."""
        return [a for a in AREAS if hasattr(getattr(data, a), name)]

    def test_the_vendored_table_agrees_with_the_datastructure_scan(self):
        """All 863 declarations that name a module land where the scan would put them.

        This is the whole argument for the swap in one assertion. `copper_rrr` is the
        single exclusion and it has its own test below.
        """
        data = DataStructure()
        for name, declaration in INPUT_VARIABLES.items():
            if declaration.module is None or name == "copper_rrr":
                continue
            assert self._scan(data, name) == [declaration.module], name

    def test_ixc_and_icc_carry_no_field_and_resolve_to_nothing(self):
        """The two `set_variable=False` rows are the problem statement, not values.

        `INPUT_VARIABLES` records them with `module=None`; a `module=None` row must not
        become `VarPath(.None.ixc)`.
        """
        assert {n for n, d in INPUT_VARIABLES.items() if d.module is None} == {
            "ixc",
            "icc",
        }
        resolve = _Resolver(_EMPTY_GRAPH)
        for name in ("ixc", "icc"):
            with pytest.raises(ValueError, match="resolves to nothing"):
                resolve(name)

    def test_copper_rrr_is_rebco_because_the_input_layer_says_so(self):
        """The one name where the table and the scan differ, and why the table wins.

        `copper_rrr` is a field on **both** `rebco` and `superconducting_tfcoil`, so the
        old scan could only raise "ambiguous". `process/core/input.py` declares it once,
        on `rebco`, and `parse_input_file` writes exactly `variable_config.module` --
        so an `IN.DAT` assignment to `copper_rrr` moves `rebco.copper_rrr` and nothing
        else, and the ambiguity was an artefact of asking `hasattr` a question only the
        input layer can answer. (`superconducting_tfcoil.copper_rrr` is a duplicate
        default: no PROCESS model reads either field.)
        """
        assert len(self._scan(DataStructure(), "copper_rrr")) == 2
        assert INPUT_VARIABLES["copper_rrr"].module == "rebco"
        assert PROCESS_INPUT_VARIABLES["copper_rrr"].module == "rebco"
        assert _Resolver(_EMPTY_GRAPH)("copper_rrr") == VarPath((
            GetAttrKey("rebco"),
            GetAttrKey("copper_rrr"),
        ))

    def test_the_non_input_table_is_exactly_the_undeclared_parameter_surface(self):
        """`NON_INPUT_FIELDS` covers every output the constraint layer can name.

        The set is *derived* here rather than transcribed, so a new `constraint_*` naming
        a quantity nobody declares as an input fails this test instead of silently
        failing to assemble on some machine nobody has run yet. One name is excluded on
        purpose: `nd_plasma_electron_max_array_7` is an array element with no field of
        its own, and the old `DataStructure` scan raised on it too.
        """
        data = DataStructure()
        names = set()
        for attribute in dir(ported_constraints):
            if attribute.startswith("constraint_"):
                names |= set(
                    inspect.signature(getattr(ported_constraints, attribute)).parameters
                )
        for metric in ported_objectives.OBJECTIVE_METRICS.values():
            names |= set(inspect.signature(metric).parameters)
        names -= set(SWITCH_PARAMETER_NAMES)
        undeclared = {
            name
            for name in names
            if INPUT_VARIABLES.get(name) is None or INPUT_VARIABLES[name].module is None
        }
        assert self._scan(data, "nd_plasma_electron_max_array_7") == []
        assert set(NON_INPUT_FIELDS) == undeclared - {"nd_plasma_electron_max_array_7"}
        assert not set(NON_INPUT_FIELDS) & {
            n for n, d in INPUT_VARIABLES.items() if d.module is not None
        }

    @pytest.mark.parametrize(("name", "area"), sorted(NON_INPUT_FIELDS.items()))
    def test_every_non_input_row_is_the_scans_own_answer(self, name, area):
        """Each row names the unique `DataStructure` area the old scan would have found.

        108 cases, and they are the whole of what the swap had to reproduce by hand:
        `INPUT_VARIABLES` could never have answered for any of them.
        """
        assert self._scan(DataStructure(), name) == [area]

    def test_the_error_message_names_both_failures_separately(self):
        """An unresolvable name is two different bugs and the message says which.

        Not in *this* graph (a configuration mismatch -- the name may be fine on another
        device) versus not a declared PROCESS input at all (a typo). The old message
        conflated them into one sentence about `DataStructure` areas.
        """
        with pytest.raises(ValueError) as excinfo:
            _Resolver(_EMPTY_GRAPH)("rmajorr")
        message = str(excinfo.value)
        assert "No node in this graph" in message
        assert "not a declared PROCESS input" in message
        assert "NON_INPUT_FIELDS" in message

    def test_a_name_the_graph_owns_still_wins_over_the_table(self):
        """Stage one is unchanged: the graph's own `VarPath`, not the input table's.

        `aspect` is a declared input (`physics`) *and* a variable of the reference
        machine; resolving it must give the graph's variable so the constraint is wired
        into the real dataflow.
        """
        assert INPUT_VARIABLES["aspect"].module == "physics"
        resolved = _Resolver(GRAPH)("aspect")
        assert resolved in GRAPH.variables

    def test_sand_imports_and_resolves_with_process_blocked(self):
        """§23.5's second open item, closed: `sand` needs no `process` at runtime.

        Proved the way `test_process_free_import.py` proves the rest -- a subprocess
        with `sys.meta_path` raising on any `process` import -- but carried here rather
        than added there, because that file was being edited by another agent when this
        was written. What it runs is the *whole* stage-two path, not just the import:
        `constraint_nodes` for the reference stellarator problem, whose 14 constraints
        send 13 distinct names to stage two.
        """
        probe = subprocess.run(  # noqa: S603
            [sys.executable, "-c", _BLOCKED_PROBE],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        assert probe.returncode == 0, probe.stderr[-4000:]
        assert "NODES 14" in probe.stdout, probe.stdout


_BLOCKED_PROBE = '''
import sys

class _Block:
    """Raise on any `process` import, from anywhere, at any depth."""
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "process" or fullname.startswith("process."):
            raise ImportError("BLOCKED: " + fullname)
        return None

sys.meta_path.insert(0, _Block())

import jax
jax.config.update("jax_enable_x64", True)

from functional_process.indat import GRAPH
from functional_process.sand import constraint_nodes

nodes, equalities, inequalities, omitted = constraint_nodes(
    GRAPH, [2, 16, 24, 8, 17, 18, 67, 82, 83, 62, 32, 34, 35, 65], 2
)
assert not omitted, omitted
assert "process" not in sys.modules, "`process` was imported despite the block"
print("NODES", len(nodes))
'''
"""Deliberately a copy of `test_process_free_import.py`'s block rather than an import of
it: a test that proves independence should not depend on another test module."""


class _Conditions:
    """The two attributes `residual_condition_scales` reads off a `Drive`.

    A stub rather than a real `Drive`: the function is pure name arithmetic over
    `unknowns`/`conditions` plus a lookup in `env`, so assembling a 130-node graph to
    exercise it would test the assembly, not the rule.
    """

    def __init__(self, unknowns, conditions):
        self.unknowns = unknowns
        self.conditions = conditions


def test_a_residual_whose_unknown_has_no_scale_keeps_a_factor_of_one():
    """`1/|u|` degrades to **`1.0`**, never to `1/floor`.

    Regression pin for a measured defect, not a hypothetical: `.power.qac` is
    identically zero on the reference run (no PF energy swing to dissipate), so
    `1 / max(|u|, 1e-12)` weighted its residual row -- the trivial equality `u = 0`,
    whose Jacobian row is exactly `-1` in one column -- by `1e12`, nine orders of
    magnitude above every other row in the problem. The condition number of the Jacobian
    `VmconDriver` hands VMCON was `6.7e12` against `2.1e4` without it, and Stage C2 went
    from 62 SQP iterations to 73 -- and, on the tree state where this was first seen, to
    `max_iter` without converging at all. `1.0` is what `VmconDriver.scaled` already
    degrades to for a design variable starting at `0.0`, so the two scalings agree.
    """
    from cottax.tools.minting import prefix_path

    from functional_process.sand import COND, residual_condition_scales

    zero = VarPath((GetAttrKey("power"), GetAttrKey("qac")))
    real = VarPath((GetAttrKey("power"), GetAttrKey("qss")))
    constraint = prefix_path(
        VarPath((GetAttrKey("constraints"), GetAttrKey("c2"))), COND
    )
    # A residualised `FixedPoint`'s condition is `^cond^cond.X` against the unknown `.X`
    # -- two mints, which `residual_condition_scales` strips to pair them.
    residual = {v: prefix_path(prefix_path(v, COND), COND) for v in (zero, real)}
    drive = _Conditions(
        unknowns=(zero, real),
        conditions=(constraint, residual[zero], residual[real]),
    )
    scales = dict(residual_condition_scales(drive, {zero: 0.0, real: 4.0}, floor=1e-12))

    # Exact comparison is the point: both factors are computed by a single division of
    # exactly representable values, and the whole defect was a factor being *silently*
    # something other than the one the rule names.
    assert scales[residual[zero]] == 1.0  # noqa: RUF069
    assert scales[residual[real]] == 0.25  # noqa: RUF069
    # PROCESS's own constraints are never touched -- that is what keeps the iterates
    # comparable with PROCESS's own.
    assert constraint not in scales


# ---------------------------------------------------------------- the driver


def _toy_problem(driver=None, objective=None):
    """`min (x-3)^2 + (y-2)^2` subject to `x + y - 4 <= 0`, as a cottax graph.

    `objective` replaces the objective's `fn` and nothing else -- one caller wants a
    body whose *derivative* is not finite at the start while every value is, which is
    the only shape `drivers._refuse_non_finite` catches that a values-only check misses.

    **Takes the driver**, because `Assign` is what mints the `^guess.*` ports this
    returns -- the algorithm's own `requires` names them, so there is nothing to hand
    back until one is chosen. Each caller passes the `VmconDriver` whose behaviour it is
    testing, where it used to pass one to `schedule_for` afterwards.

    Written out rather than reusing the real graph so the answer is known in closed
    form: the unconstrained minimum `(3, 2)` violates the constraint, so it is active and
    the solution is the projection `(2.5, 1.5)` with objective `0.5`. That makes this a
    real test of the **sign convention** -- with the inequality passed unflipped, VMCON
    would read the feasible set as `x + y >= 4` and stop at `(3, 2)`.
    """
    x = VarPath((GetAttrKey("d"), GetAttrKey("x")))
    y = VarPath((GetAttrKey("d"), GetAttrKey("y")))
    f = VarPath((GetAttrKey("c"), GetAttrKey("f")))
    g = VarPath((GetAttrKey("c"), GetAttrKey("g")))
    graph = Graph(
        path_map([
            (
                NodePath((GetAttrKey("F"),)),
                CallableNode(
                    inputs=(In(x), In(y)),
                    outputs=(Out(f),),
                    fn=_Objective() if objective is None else objective,
                ),
            ),
            (
                NodePath((GetAttrKey("G"),)),
                CallableNode(
                    inputs=(In(x), In(y)),
                    outputs=(Out(g),),
                    fn=_Constraint(),
                ),
            ),
            (
                NodePath((GetAttrKey("Opt"),)),
                Optimise(
                    objective=In(f),
                    design=(Out(x), Out(y)),
                    inequalities=(In(g),),
                ),
            ),
        ])
    )
    # Every problem this port drives declares its `Start`s (`mda.driven_graph` and
    # `sand.optimise_graph` both do it); `VmconDriver.requires` names `Start`, so the
    # toy problem has to say so too or `Drive` refuses the pair.
    problem = NodePath((GetAttrKey("Opt"),))
    if driver is None:
        driver = VmconDriver(n_equality=0, n_inequality=1, scaled=False)
    graph = Assign(problem, driver).apply(graph)
    gx, gy = driver_vars(graph[problem], Start)
    return graph, x, y, gx, gy


class _Objective(jax.tree_util.register_static and object):
    """`(x-3)^2 + (y-2)^2`. A class, not a lambda, so the node definition stays a stable
    jit cache key -- the same reason `cottax.rewrites.Compare` uses `Pairwise`.
    """

    def __call__(self, x, y):
        return (x - 3.0) ** 2 + (y - 2.0) ** 2

    def __eq__(self, other):
        return isinstance(other, _Objective)

    def __hash__(self):
        return hash(type(self))


class _Constraint:
    """`x + y - 4`, in cottax's `g <= 0` convention."""

    def __call__(self, x, y):
        return x + y - 4.0

    def __eq__(self, other):
        return isinstance(other, _Constraint)

    def __hash__(self):
        return hash(type(self))


def test_vmcon_driver_reaches_a_known_constrained_optimum():
    graph, x, y, gx, gy = _toy_problem()
    graph, x, y, gx, gy = _toy_problem(
        VmconDriver(n_equality=0, n_inequality=1, scaled=False)
    )
    schedule = Schedule(Blocking.scc(graph))
    out = schedule.run(path_map({gx: jnp.asarray(0.0), gy: jnp.asarray(0.0)}))
    assert float(out[x]) == pytest.approx(2.5, abs=1e-6)
    assert float(out[y]) == pytest.approx(1.5, abs=1e-6)


def test_vmcon_driver_honours_bounds_as_bounds():
    """A box bound is handed to VMCON as a bound, not re-expressed as two inequality
    constraints -- which would be a different QP subproblem and different iterates.
    """
    graph, x, y, gx, gy = _toy_problem()
    graph, x, y, gx, gy = _toy_problem(
        VmconDriver(
            n_equality=0,
            n_inequality=1,
            scaled=False,
            bounds=((x, -np.inf, 2.0),),
        )
    )
    schedule = Schedule(Blocking.scc(graph))
    out = schedule.run(path_map({gx: jnp.asarray(0.0), gy: jnp.asarray(0.0)}))
    assert float(out[x]) <= 2.0 + 1e-9


def test_vmcon_driver_scaling_does_not_move_the_answer():
    """PROCESS's `x * (1/x_start)` conditioning changes the path, never the optimum."""
    graph, x, y, gx, gy = _toy_problem()
    graph, x, y, gx, gy = _toy_problem(
        VmconDriver(n_equality=0, n_inequality=1, scaled=True)
    )
    out = Schedule(Blocking.scc(graph)).run(
        path_map({gx: jnp.asarray(1.0), gy: jnp.asarray(1.0)})
    )
    assert float(out[x]) == pytest.approx(2.5, abs=1e-6)
    assert float(out[y]) == pytest.approx(1.5, abs=1e-6)


def test_vmcon_driver_refuses_a_wrong_condition_count():
    """`ConditionMap` cannot say which condition is which, so the counts are a contract
    the driver is given -- and a stale pair must fail loudly, not mislabel a constraint.
    """
    graph, x, y, gx, gy = _toy_problem()
    graph, x, y, gx, gy = _toy_problem(
        VmconDriver(n_equality=1, n_inequality=1, scaled=False)
    )
    schedule = Schedule(Blocking.scc(graph))
    with pytest.raises(ValueError, match="equalities"):
        schedule.run(path_map({gx: jnp.asarray(0.0), gy: jnp.asarray(0.0)}))


class _UnboundedSlope:
    """`sqrt(x) + y`: finite everywhere on `x >= 0`, with an infinite slope at `x = 0`.

    The shape `drivers._refuse_non_finite`'s docstring names as the usual cause -- a
    value that is perfectly well defined and a derivative that is not -- so a start at
    `x = 0` produces exactly the failure a values-only probe cannot see.
    """

    def __call__(self, x, y):
        return jnp.sqrt(x) + y

    def __eq__(self, other):
        return isinstance(other, _UnboundedSlope)

    def __hash__(self):
        return hash(type(self))


TOY_PROBLEM = NodePath((GetAttrKey("Opt"),))
"""Where `_toy_problem` binds its `Optimise`, which is the place every `DriverOut` port
it mints is named from (`^driver_out.<label>.Opt`)."""


def _non_finite_toy():
    """The toy problem with `sqrt(x)` as its objective, started at `x = 0`."""
    graph, _x, _y, gx, gy = _toy_problem(
        VmconDriver(n_equality=0, n_inequality=1, scaled=False),
        objective=_UnboundedSlope(),
    )
    return graph, path_map({gx: jnp.asarray(0.0), gy: jnp.asarray(1.0)})


def test_the_non_finite_refusal_carries_its_lists_and_a_one_line_summary():
    """`_refuse_non_finite`'s own shape, asked of it directly -- the eager path.

    Raising is the right failure for a **direct** call, and the named type is what a
    caller tells this apart by: a solve raises `ValueError` for a stale condition count,
    a stray `condition_scale` name and an inert objective, and catching `ValueError`
    would label all of them "non-finite".

    The **separation of value from derivative** is the load-bearing part and is checked
    on a point where every value is finite and one row's derivative is not -- the exact
    case the deleted pre-solve probe could not see, and the one that cost a full
    investigation.
    """
    f = VarPath((GetAttrKey("c"), GetAttrKey("f")))
    g = VarPath((GetAttrKey("c"), GetAttrKey("g")))
    x = VarPath((GetAttrKey("d"), GetAttrKey("x")))

    class _Map:
        conditions = (f, g)
        unknowns = (x,)

    with pytest.raises(NonFiniteProblemError) as raised:
        _refuse_non_finite(np.array([1.0, 2.0]), np.array([[np.nan], [1.0]]), _Map())
    refusal = raised.value
    assert isinstance(refusal, ValueError)  # every existing `except` still catches it
    assert refusal.bad_values == ()
    assert refusal.bad_rows == (f.path_str(),)
    assert refusal.zero_columns == ()
    assert refusal.n_conditions == 2
    assert "0/2 non-finite in VALUE" in refusal.summary
    assert f"1/2 non-finite in DERIVATIVE ({f.path_str()})" in refusal.summary


@pytest.mark.parametrize("whole", [False, None])
def test_a_non_finite_solve_returns_a_status_and_raises_nothing(whole):
    """**No exception crosses the `jax.pure_callback` boundary. This is the contract.**

    `pure_callback` promises a pure function of its inputs: jax may elide it under DCE,
    run it more than once, or reorder it, so raising out of one is a side effect and
    what a *compiled* callback does with a Python exception is unspecified (measured: it
    arrives as `JaxRuntimeError` carrying the traceback as text, class and attributes
    gone). Every path in this tree runs a `VmconDriver` inside a compiled callback --
    `cottax`'s own `Schedule.run` and `sand_harness.run_schedule` alike, `whole=False`
    included, which is what `run_cold_matrix.cold_sand` calls -- so the refusal must not
    travel that way.

    It does not: `host` catches it inside the callback and reports `VMCON_NON_FINITE`
    through the driver's `Status` port, which lands in the returned env like every other
    value. The run **completes**; the caller reads a verdict.

    Both `whole` settings are checked because they choose different jit strategies and
    the shape of the outcome must not depend on which one a schedule got.
    """
    graph, env = _non_finite_toy()
    schedule = Schedule(Blocking.scc(graph))
    # `run_schedule` takes a plain mapping (`_inputs_only`'s shape), `Schedule.run` a
    # `PathMap`; the two jit strategies are the point of the parametrisation, not the
    # container.
    out = (
        run_schedule(schedule, dict(env.items()), whole=False)
        if whole is False
        else schedule.run(env)
    )
    status = out[Status.name_for(TOY_PROBLEM)]
    assert int(np.asarray(status)) == VMCON_NON_FINITE
    assert not bool(np.asarray(out[Converged.name_for(TOY_PROBLEM)]))
    # No step was taken, so the reported point is the start, not a "best" one.
    assert int(np.asarray(out[Steps.name_for(TOY_PROBLEM)])) == 0


def test_the_status_is_turned_back_into_names_by_non_finite_summary():
    """The other half of the split: the status says *that*, this says *what*.

    Recomputed on the host from the same `_refuse_non_finite`, so there is one
    definition of "non-finite problem" in the module and the reporting cannot drift from
    the refusing. `None` on a healthy point, so a caller can use it as the test as well
    as the message.
    """
    graph, env = _non_finite_toy()
    drive = next(
        step for step in Schedule(Blocking.scc(graph)).steps if hasattr(step, "problem")
    )
    context = {v: env[v] for v in drive.context if v in env}
    conditions = drive.condition_map(context)
    # The env holds the starts at their `^guess.*` ports; the unknowns are the bare
    # names, so the start for each is looked up by that correspondence.
    guesses = {v.path_str().removeprefix("^guess"): env[v] for v in env.keys()}
    bad, unravel = ravel_pytree(
        tuple(jnp.asarray(guesses[u.path_str()]) for u in conditions.unknowns)
    )
    summary = non_finite_summary(conditions, unravel, bad)
    assert summary is not None
    assert "non-finite in DERIVATIVE" in summary
    assert "0/2 non-finite in VALUE" in summary


# ------------------------------------------------- what the driver says about its run


def test_vmcon_driver_returns_its_reports_after_the_unknowns():
    """`AbstractDriver.__call__`'s contract: the unknowns, then one value per kind in
    `reports`, in `reports` order.

    Asked of the **driver**, directly, so this pins the contract at the seam rather than
    at any one caller's re-implementation of it -- `Drive.__call__`, `mdf.solve` and
    `sand_harness._driven_runner` all read the same positional split, and the last of
    the three had it wrong (see `test_driven_runner_binds_reports_not_unknowns`).
    """
    driver = VmconDriver(n_equality=0, n_inequality=1, scaled=False)
    graph, x, y, gx, gy = _toy_problem(driver)
    drive = next(
        step for step in Schedule(Blocking.scc(graph)).steps if isinstance(step, Drive)
    )
    answered = driver(
        drive.condition_map({}), {Start: (jnp.asarray(0.0), jnp.asarray(0.0))}
    )
    assert len(answered) == len(drive.unknowns) + len(driver.reports)
    values, (steps, converged, status) = answered[:2], answered[2:]
    assert float(values[0]) == pytest.approx(2.5, abs=1e-6)
    assert float(values[1]) == pytest.approx(1.5, abs=1e-6)
    assert int(steps) > 0
    assert bool(converged) is True
    assert int(status) == VMCON_CONVERGED


def test_vmcon_driver_reports_which_pyvmcon_exception_fired():
    """A cap hit is `Converged False` with the base-class `Status`, and the driver still
    returns the best point rather than raising -- `solver.py:262-272`'s own pattern.

    The three `VMCON_STATUS` codes are the whole reason `Status` is not a bool: a solve
    that ran out of iterations and one whose first QP had no feasible point are opposite
    diagnoses and used to be distinguishable only by which exception a caller's
    `except` clause happened to see.
    """
    driver = VmconDriver(n_equality=0, n_inequality=1, scaled=False, max_iter=1)
    graph, x, y, gx, gy = _toy_problem(driver)
    drive = next(
        step for step in Schedule(Blocking.scc(graph)).steps if isinstance(step, Drive)
    )
    *_values, steps, converged, status = driver(
        drive.condition_map({}), {Start: (jnp.asarray(0.0), jnp.asarray(0.0))}
    )
    assert bool(converged) is False
    assert int(status) == VMCON_STATUS["VMCONConvergenceException"]
    assert int(steps) == 1


def test_driven_runner_binds_reports_not_unknowns():
    """`sand_harness._driven_runner` must bind `unknowns + reports`, as `Drive` does.

    **The regression this exists for.** `_driven_runner` re-implements
    `cottax.evaluate.Drive.__call__` so that the driver can stay eager while the body is
    jitted, and its binding read

        if len(converged) != len(step.unknowns): raise ...
        env.update(zip(step.unknowns, converged, strict=True))

    -- the unknowns alone. That was invisible for as long as every driver reaching it
    reported nothing, and became a spurious `ValueError` the moment `VmconDriver` gained
    `(Steps, Converged, Status)`. It could *not* have mis-bound a verdict to a design
    variable, which is the failure one expects here: reports come after the unknowns
    positionally, so even a truncating `zip` pairs each unknown with its own value and
    merely drops the verdict. Both halves are asserted below anyway, because the order is
    a contract and not a guarantee this test should assume.

    Checked on the **walk** path (`whole=False`), because that is the path
    `_driven_runner` is on: the whole-schedule jit goes through cottax's own `Drive` and
    could never have had this bug.
    """
    driver = VmconDriver(n_equality=0, n_inequality=1, scaled=False)
    graph, x, y, gx, gy = _toy_problem(driver)
    schedule = Schedule(Blocking.scc(graph))
    out = run_schedule(
        schedule, {gx: jnp.asarray(0.0), gy: jnp.asarray(0.0)}, whole=False
    )
    # The unknowns hold the optimum and not a verdict -- the half a non-strict `zip`
    # would have got wrong silently.
    assert float(out[x]) == pytest.approx(2.5, abs=1e-6)
    assert float(out[y]) == pytest.approx(1.5, abs=1e-6)
    # ...and the verdict is in the env under the names `Assign` minted for it.
    assert int(out[Steps.name_for(TOY_PROBLEM)]) > 0
    assert bool(out[Converged.name_for(TOY_PROBLEM)]) is True
    assert int(out[Status.name_for(TOY_PROBLEM)]) == VMCON_CONVERGED


def test_driven_runner_and_whole_jit_agree_on_the_verdict():
    """The two `run_schedule` paths bind the same reports to the same names.

    They are different code -- cottax's `Drive.__call__` under one jit, this port's
    `_driven_runner` walking the steps -- and the point of the `DriverOut` change is
    that a caller reads the verdict the same way whichever ran.
    """
    driver = VmconDriver(n_equality=0, n_inequality=1, scaled=False)
    graph, x, y, gx, gy = _toy_problem(driver)
    schedule = Schedule(Blocking.scc(graph))
    env = {gx: jnp.asarray(0.0), gy: jnp.asarray(0.0)}
    walked = run_schedule(schedule, dict(env), whole=False)
    jitted = Schedule(Blocking.scc(graph)).run(path_map(dict(env)))
    for var in (x, y, Steps.name_for(TOY_PROBLEM), Status.name_for(TOY_PROBLEM)):
        # Exact, deliberately: the claim is that the two paths bind the *same*
        # value, not a nearby one.
        assert float(np.asarray(walked[var])) == float(  # noqa: RUF069
            np.asarray(jitted[var])
        )


def test_vmcon_driver_traces_under_one_whole_schedule_jit():
    """The `pure_callback` wrap's whole purpose: a `Drive` answered by `VmconDriver` is
    now inside one traced program instead of being a hole in it.

    Before the wrap this raised `TracerArrayConversionError` on the unknowns vector --
    `design_scale` and the scaled bounds read the *values* of a start that is a tracer
    under a trace -- and `run_schedule` recorded that refusal and fell back to the walk.
    `schedule_verdict` is where that decision is readable, so it is what is asserted:
    a regression here is silent otherwise, costing only speed and structure.
    """
    import functional_process.sand_harness as harness

    driver = VmconDriver(n_equality=0, n_inequality=1, scaled=False)
    graph, x, y, gx, gy = _toy_problem(driver)
    schedule = Schedule(Blocking.scc(graph))
    # `run_schedule` caches its verdict per schedule, and an equal schedule is the same
    # key -- the tests above deliberately run this very one with `whole=False`, which
    # records `False` with no reason. Evicting is the honest way to ask the question;
    # perturbing a driver field to get a different hash would be testing a different
    # schedule and calling it this one.
    harness._SCHEDULE_WHOLE.pop(schedule, None)
    harness._SCHEDULE_VERDICT.pop(schedule, None)
    out = run_schedule(schedule, {gx: jnp.asarray(0.0), gy: jnp.asarray(0.0)})
    assert schedule_verdict(schedule) == (True, None)
    assert float(out[x]) == pytest.approx(2.5, abs=1e-6)


def test_vmcon_driver_answer_survives_the_callback_boundary_bitwise():
    """`jax.pure_callback` canonicalises dtypes at its boundary, and this problem is
    sensitive at the last bit (`_audit/optimise_design.md` §21.2), so the wrap converts
    to `float64` explicitly on the way in and back to the start's dtype on the way out.

    Bitwise, not `approx`: a tolerance here would pass on exactly the drift the explicit
    conversion exists to prevent. The two paths differ in whether the solve runs inside
    a trace, and nothing else.
    """
    driver = VmconDriver(n_equality=0, n_inequality=1, scaled=True)
    graph, x, y, gx, gy = _toy_problem(driver)
    schedule = Schedule(Blocking.scc(graph))
    env = {gx: jnp.asarray(1.0), gy: jnp.asarray(1.0)}
    walked = run_schedule(schedule, dict(env), whole=False)
    jitted = jax.jit(
        lambda values: path_map(
            Schedule(Blocking.scc(graph)).run(path_map(dict(values)))
        )
    )(path_map(env))
    jitted = dict(jitted)
    assert float(walked[x]) == float(jitted[x])  # noqa: RUF069 -- bitwise is the point
    assert float(walked[y]) == float(jitted[y])  # noqa: RUF069 -- bitwise is the point


def test_slsqp_driver_reports_scipys_own_alphabet():
    """`SlsqpDriver` reports the same three kinds, in scipy's numbering.

    The kinds are shared and the *codes* are not: `Status` is port-local precisely
    because `pyvmcon`'s three exceptions and `scipy`'s `OptimizeResult.status` are
    different alphabets, and a caller comparing the two SQPs has to know which it is
    reading. `0` is scipy's own "terminated successfully".
    """
    driver = SlsqpDriver(n_equality=0, n_inequality=1, scaled=False)
    graph, x, y, gx, gy = _toy_problem(driver)
    drive = next(
        step for step in Schedule(Blocking.scc(graph)).steps if isinstance(step, Drive)
    )
    *values, steps, converged, status = driver(
        drive.condition_map({}), {Start: (jnp.asarray(0.0), jnp.asarray(0.0))}
    )
    assert float(values[0]) == pytest.approx(2.5, abs=1e-6)
    assert bool(converged) is True
    assert int(status) == 0
    assert int(steps) > 0


def test_outcome_sink_is_gone():
    """The mutable results sink both SQP drivers used to write into is deleted.

    A `DriverOut` port and an `Outcome` dict are two channels for one fact, and keeping
    both is how they drift apart. This asserts the deletion rather than trusting a grep:
    a re-added field would restore a hashability trap (`Schedule.__hash__` reaches a
    driver's fields) that cost a whole investigation once already.
    """
    import functional_process.core.solver.drivers as drivers_module

    assert not hasattr(drivers_module, "Outcome")
    assert "outcome" not in SlsqpDriver.__dataclass_fields__
    assert "outcome" not in VmconDriver.__dataclass_fields__


# ---------------------------------------------------------------- the tokamak study

TOKAMAK_IXC = [4, 6]
TOKAMAK_ICC = [
    1, 2, 5, 8, 9, 13, 15, 30, 16, 24, 25, 26, 27, 33, 34, 35, 36, 60, 62, 65,
    72, 81, 68, 31, 32,
]  # fmt: skip
TOKAMAK_N_EQUALITY = 2
TOKAMAK_FIGURE_OF_MERIT = 7
"""`large_tokamak_eval.IN.DAT`'s own problem: an evaluation-style file -- two design
variables against two equalities (`i_process_run_mode = -2`, so PROCESS itself answers
it with `fsolve` over the equalities and merely reports the 23 inequalities).
`i_figure_merit` is not in the file at all: 7 (capital cost) is `NumericsData`'s own
default, and both facts are checked below rather than trusted."""


import functools  # noqa: E402


@functools.lru_cache(maxsize=1)
def _tokamak_graph():
    from functional_process.boundary import TOKAMAK_INPUT_FILE
    from functional_process.indat import graph_for, machine_from_indat

    return graph_for(machine_from_indat(str(REPO_ROOT / TOKAMAK_INPUT_FILE)))


def test_tokamak_problem_matches_the_input_file():
    """The constants above really are what the tokamak `IN.DAT` asks for.

    Same discipline as `test_reference_problem_matches_the_input_file`, plus the parts
    that file makes parseable: its `icc`/`ixc` are one id per line, so both lists are
    read outright rather than only the shape scalars.
    """
    from functional_process.boundary import TOKAMAK_INPUT_FILE

    text = (REPO_ROOT / TOKAMAK_INPUT_FILE).read_text()
    settings = {}
    icc, ixc = [], []
    for line in text.splitlines():
        if m := re.match(r"\s*icc\s*=\s*(\d+)", line):
            icc.append(int(m.group(1)))
        elif m := re.match(r"\s*ixc\s*=\s*(\d+)", line):
            ixc.append(int(m.group(1)))
        elif m := re.match(r"\s*([A-Za-z_]\w*)\s*=\s*(-?\d+)\s*(\*.*)?$", line):
            settings[m.group(1)] = int(m.group(2))
    assert icc == TOKAMAK_ICC
    assert ixc == TOKAMAK_IXC
    assert settings["n_equality_constraints"] == TOKAMAK_N_EQUALITY
    assert settings["i_process_run_mode"] == -2
    # The figure of merit is PROCESS's default, not the file's -- so the check is that
    # the file is silent and the default is what the constant says.
    assert "i_figure_merit" not in settings
    from process.data_structure.numerics import NumericsData

    assert NumericsData().i_figure_merit == TOKAMAK_FIGURE_OF_MERIT


def test_tokamak_switch_parameters_match_the_contracts_static_argnames():
    """`SWITCH_PARAMETER_NAMES` decides, mechanically, which parameters
    `switch_values_for` freezes for the tokamak's active constraints -- and the same
    two-directional drift check as the stellarator's applies: a switch not in the tuple
    would be traced onto a port, a real variable in the tuple would be frozen where a
    derivative belongs.
    """
    from functional_process.sand import SWITCH_PARAMETER_NAMES

    contracts = _contract_static_argnames()
    for cid in TOKAMAK_ICC:
        fn = getattr(ported_constraints, f"constraint_{cid}")
        parameters = set(inspect.signature(fn).parameters)
        assert parameters & set(SWITCH_PARAMETER_NAMES) == set(contracts[cid]), (
            f"constraint {cid}: `switch_values_for` would bind "
            f"{sorted(parameters & set(SWITCH_PARAMETER_NAMES))} statically, but its "
            f"Tier1Contract declares static_argnames={contracts[cid]}"
        )


def test_switch_values_for_reads_the_active_surface_only():
    """One value per switch actually in an active signature, every value an `int` --
    read off a `DataStructure` with no PROCESS run behind it, which is exactly the
    point: no default is transcribed into the port.
    """
    from functional_process.sand import switch_values_for
    from process.core.model import DataStructure

    values = switch_values_for(DataStructure(), TOKAMAK_ICC, TOKAMAK_FIGURE_OF_MERIT)
    assert set(values) == {
        "i_rad_loss",
        "i_plasma_ignited",
        "i_beta_component",
        "istell",
        "i_density_limit",
        "i_tf_bucking",
        "i_tf_inside_cs",
        "i_q95_fixed",
        "ireactor",  # the objective's (capital cost branches on it)
    }
    assert all(isinstance(v, int) for v in values.values())


def test_tokamak_design_variables_are_boundary_inputs():
    """The `Optimise` owning `temp_plasma_electron_vol_avg_kev` and
    `nd_plasma_electrons_vol_avg` collides with nothing in the tokamak graph -- the same
    cheapness argument as the stellarator's eight.
    """
    owned = set(_tokamak_graph().owners)
    for i in TOKAMAK_IXC:
        assert iteration_variable_path(i) not in owned


def test_the_pf_ring_is_detected_as_an_array_unknown_problem():
    """`sand.array_valued_problems` finds the tokamak's PF-coil cycle -- the one
    `FixedPoint` whose unknowns are arrays (`ind_pf_cs_plasma_mutual`,
    `n_pf_coil_turns`) -- **at the env's own values**, not from a name list. The
    scalar SAND layer cannot absorb it (`array_valued_problems`' docstring names each
    seam), so `sand_harness.assemble` drops and reports it; this pins the detection
    the report rests on. With every unknown seeded scalar the same problem is *not*
    flagged, which is the "detected, not listed" half.
    """
    from cottax.problem import FixedPoint as CottaxFixedPoint

    from functional_process.mda import cut_graph
    from functional_process.mda_harness import _without_excluded
    from functional_process.sand import array_valued_problems

    graph = cut_graph(_without_excluded(_tokamak_graph()))
    pf = [
        p
        for p in graph.declared
        if isinstance(graph[p], CottaxFixedPoint)
        and any("pf_coil" in u.path_str() for u in graph[p].owns)
    ]
    assert len(pf) == 1
    unknowns = graph[pf[0]].owns
    arrays = {u: jnp.ones((22,)) for u in unknowns}
    assert array_valued_problems(graph, arrays) == (pf[0],)
    scalars = {u: jnp.asarray(1.0) for u in unknowns}
    assert array_valued_problems(graph, scalars) == ()
    # An unknown the env has no value for cannot be measured, so nothing is flagged.
    assert array_valued_problems(graph, {}) == ()


def test_tokamak_sand_assembles_and_orders_its_conditions():
    """The whole tokamak assembly: all 25 constraints resolve, c1/c2 are the
    equalities, the objective leads the conditions, and the two design variables lead
    the unknowns. The stellarator analogue is
    `test_sand_assembles_and_orders_its_conditions`; the assertions are the same ones
    because the contract is.
    """
    from functional_process.mda import cut_graph
    from functional_process.mda_harness import _without_excluded
    from functional_process.sand import switch_values_for
    from process.core.model import DataStructure

    switch_values = switch_values_for(
        DataStructure(), TOKAMAK_ICC, TOKAMAK_FIGURE_OF_MERIT
    )
    graph = cut_graph(_without_excluded(_tokamak_graph()))
    with_problem, _name, report = optimise_graph(
        graph,
        TOKAMAK_IXC,
        TOKAMAK_ICC,
        TOKAMAK_N_EQUALITY,
        TOKAMAK_FIGURE_OF_MERIT,
        switch_values=switch_values,
    )
    assert report["omitted"] == {}
    assert [v.path_str() for v in report["equalities"]] == [
        "^cond.constraints.c1",
        "^cond.constraints.c2",
    ]
    assert len(report["inequalities"]) == len(TOKAMAK_ICC) - TOKAMAK_N_EQUALITY
    combined, _residualised = sand_graph(with_problem)
    schedule = sand_schedule(combined, None)
    shape = sand_shape(schedule)
    drive = shape["drive"]
    names = [c.path_str() for c in drive.conditions]
    assert names[0] == "^cond.numerics.objf"
    definition = drive.subgraph[drive.problem].problem
    assert names[1 : 1 + len(definition.equalities)] == [
        c.var.path_str() for c in definition.equalities
    ]
    assert [v.path_str() for v in drive.unknowns[: len(TOKAMAK_IXC)]] == [
        iteration_variable_path(i).path_str() for i in TOKAMAK_IXC
    ]
    assert shape["drive_nodes"] < len(combined.nodes)
    assert shape["schedule_steps"] > 1


def test_vmcon_driver_needs_a_start():
    """No start supplied is an error, not a silent default -- there is no shape to
    guess a pytree of unknowns from.

    Spelled as an empty driver-data mapping: `Drive.role_data` builds that mapping from
    the driver's `requires`, so calling the driver directly with `{}` is what "no start"
    now looks like (the old `start=None` positional is gone).
    """
    driver = VmconDriver(n_equality=0, n_inequality=1)
    graph, x, y, gx, gy = _toy_problem(driver)
    (drive,) = Schedule(Blocking.scc(graph)).steps
    with pytest.raises(ValueError, match="starting value"):
        driver(drive.condition_map({}), {})


# ---------------------------------------------------- the degeneracy instrument


class _Scale:
    """`y = factor * x`. A class, not a lambda, for `_Objective`'s reason -- a stable
    jit cache key.
    """

    def __init__(self, factor):
        self.factor = factor

    def __call__(self, x):
        return self.factor * x

    def __eq__(self, other):
        return isinstance(other, _Scale) and other.factor == self.factor

    def __hash__(self):
        return hash((type(self), self.factor))


def _chain_fixed_point(second):
    """A `FixedPoint` whose `g` is computed by **two** nodes: `g(u) = second * 3 * u`.

    The shape the repaired `fixed_point_residuals` exists for and the old one could not
    see: the condition's direct producer (`B`) is not the whole body, because `B` reads
    `.toy.a`, which `A` produces from the unknown. A one-level body either loses the `A`
    edge or cannot run at all, and either way the measured `dg/du` is wrong.
    """
    u = VarPath((GetAttrKey("toy"), GetAttrKey("u")))
    a = VarPath((GetAttrKey("toy"), GetAttrKey("a")))
    hat = VarPath((GetAttrKey("^hat"), GetAttrKey("toy"), GetAttrKey("u")))
    problem = NodePath((GetAttrKey("P"),))
    graph = Graph(
        path_map([
            (
                NodePath((GetAttrKey("A"),)),
                CallableNode(inputs=(In(u),), outputs=(Out(a),), fn=_Scale(3.0)),
            ),
            (
                NodePath((GetAttrKey("B"),)),
                CallableNode(inputs=(In(a),), outputs=(Out(hat),), fn=_Scale(second)),
            ),
            (problem, FixedPoint(inputs=(In(hat),), outputs=(Out(u),))),
        ])
    )
    env = {
        u: jnp.asarray(2.0),
        a: jnp.asarray(6.0),
        hat: jnp.asarray(6.0 * second),
    }
    return graph, problem, env


def test_a_multi_node_cycles_residual_is_measured_and_not_guessed():
    """`d(g(u) - u)/du` on a two-node body is `3 * second - 1`, exactly.

    The regression guard for `_audit/optimise_design.md` §16.7: the body used to be the
    conditions' **direct** producers, which here is `B` alone, so `A`'s edge from the
    unknown was either missing (the derivative then comes out `-1` for every `second`,
    i.e. always "healthy") or unrunnable (`_run_acyclic` raises, and the old bare
    `except Exception: continue` filed the block as healthy too). Both failures are
    caught by asserting the *number*, which is why this checks the Jacobian and not
    merely that the call returned.
    """
    from functional_process.sand import fixed_point_residuals

    for second, expected in ((0.25, 3.0 * 0.25 - 1.0), (2.0, 3.0 * 2.0 - 1.0)):
        graph, problem, env = _chain_fixed_point(second)
        (measured,) = fixed_point_residuals(graph, env)
        assert measured.problem == problem
        assert measured.undetectable is None
        assert measured.jacobian == pytest.approx(np.array([[expected]]))
        assert measured.rank == 1
        assert measured.columns == 1
        assert not measured.degenerate


def test_an_identity_two_node_cycle_is_detected_as_degenerate():
    """`second = 1/3` makes `g(u) = u` -- an identity fixed point behind two nodes.

    The case the old one-level body could not reach at all: `EtaTurbineStep` and
    `CplifeAvail`, the two it was written for, were single-node cycles, so an identity
    hidden behind a longer chain was reported healthy and would reach `reduce_jacobian`
    as a zero row of `J_RY`, i.e. a singular equality block.
    """
    from functional_process.sand import degenerate_fixed_points, fixed_point_residuals

    graph, problem, env = _chain_fixed_point(1.0 / 3.0)
    (measured,) = fixed_point_residuals(graph, env)
    assert measured.degenerate
    assert measured.rank == 0
    assert degenerate_fixed_points(graph, env) == (problem,)


def test_an_unmeasurable_block_is_reported_rather_than_called_healthy():
    """An env with no value for a body input cannot be differentiated -- and that must
    not come back as "not degenerate".

    The narrowing of the bare `except`: `fixed_point_residuals` records the exception
    against the block, and `degenerate_fixed_points` refuses to answer for **any** block
    rather than reporting the rest healthy. Its own docstring records that this `except`
    has now silently reported every fixed point healthy twice.
    """
    from functional_process.sand import degenerate_fixed_points, fixed_point_residuals

    graph, _problem, env = _chain_fixed_point(0.25)
    starved = {k: v for k, v in env.items() if k.path_str() != ".toy.u"}
    (measured,) = fixed_point_residuals(graph, starved)
    assert measured.jacobian is None
    assert measured.rank is None
    assert not measured.degenerate
    assert "KeyError" in measured.undetectable
    with pytest.raises(ValueError, match="cannot tell whether"):
        degenerate_fixed_points(graph, starved)


def test_an_array_valued_fixed_point_is_still_measurable():
    """The tokamak PF ring owns arrays (see
    `test_the_pf_ring_is_detected_as_an_array_unknown_problem`), and the old
    `jnp.stack(...).reshape(len(reads))` raised on exactly those -- one more block the
    bare `except` filed as healthy. Flattening makes the residual a real `(n, n)` matrix,
    so its rank is a number rather than an exception.
    """
    from functional_process.sand import fixed_point_residuals

    u = VarPath((GetAttrKey("toy"), GetAttrKey("v")))
    hat = VarPath((GetAttrKey("^hat"), GetAttrKey("toy"), GetAttrKey("v")))
    problem = NodePath((GetAttrKey("P"),))
    graph = Graph(
        path_map([
            (
                NodePath((GetAttrKey("A"),)),
                CallableNode(inputs=(In(u),), outputs=(Out(hat),), fn=_Scale(0.5)),
            ),
            (problem, FixedPoint(inputs=(In(hat),), outputs=(Out(u),))),
        ])
    )
    env = {u: jnp.ones((4,)), hat: jnp.full((4,), 0.5)}
    (measured,) = fixed_point_residuals(graph, env)
    assert measured.undetectable is None
    assert measured.jacobian.shape == (4, 4)
    assert measured.jacobian == pytest.approx(-0.5 * np.eye(4))
    assert measured.rank == 4


def test_boundary_seeds_agree_with_guess_sources():
    """`cottax.boundary.seeds` and `mda.guess_sources` name the same start values.

    The two answer "where does this `^guess.*` port's value come from" differently:
    `seeds` says `unminted(port)` -- the place in a caller's own structure -- and
    `guess_sources` says the *unknown*, which for a `FixedPointCut` is the minted copy
    `^hat.X`. `sand_harness.mda_env` seeds by `guess_sources`, so this pins the claim
    its comment makes: wherever the two disagree by name, `ground_truth` still reaches
    the same `DataStructure` field, because it falls back to `unminted` for a mint with
    no `KNOWN_MINT_VALUES` entry and no entry is spelled `^hat.*`.

    A *missing* start raises in either shape and is the loud case; a start pointed at a
    different place is silent, which is what this test exists for.
    """
    from cottax.boundary import seeds
    from cottax.tools.minting import unminted

    from functional_process.mda import guess_sources
    from functional_process.mda_harness import KNOWN_MINT_VALUES
    from functional_process.sand_harness import mda_schedule

    _driven, runnable, schedule, _run = mda_schedule(None)
    guesses = guess_sources(runnable)
    seeded = seeds(schedule)

    assert set(seeded) == {v for v in schedule.inputs if v in guesses}
    for port, place in seeded.items():
        assert unminted(guesses[port]) == place, (
            f"{port.path_str()} is seeded from {place.path_str()} by the boundary and "
            f"from {guesses[port].path_str()} here, and they are not the same place"
        )
        assert guesses[port].path_str() not in KNOWN_MINT_VALUES, (
            f"{guesses[port].path_str()} now has an analytic mint value, so "
            f"`ground_truth` no longer agrees with the boundary's `unminted` seed"
        )
