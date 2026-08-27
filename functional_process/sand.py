"""The `Optimise` layer: PROCESS's own optimisation problem, assembled onto the graph
and solved as **SAND** (Simultaneous ANalysis and Design).

`mda.py` turns `indat.GRAPH` into something that can be *run*. This module turns
it into something that can be *solved*: it registers one `CallableNode` per active
constraint and one for the objective (each owning a minted `^cond.*`), inserts an
`Optimise` `DeclaredNode` owning the active iteration variables' `VarPath`s, then
`Residualise`s every remaining `FixedPoint` and `Combine`s every problem into a single
`^problem.sand` that `core.solver.drivers.VmconDriver` answers.

The design and the experiments behind every choice here are `_audit/optimise_design.md`;
this docstring records only what a reader of the code needs.

Why SAND and not MDF
--------------------
The obvious shape -- an `Optimise` *outside* the graph's 9 driven MDA blocks -- is not
expressible. Adding an `Optimise` fuses every driven SCC into one giant component with
two or more declared problems in it, and cottax refuses that in three independent places
(`Blocking.problem_types`, `schedule_for`'s driver resolution, `Drive.__check_init__`
via `Graph.problem_type`). `Blocking.nest` *records* the nesting correctly but
`evaluate.py` never reads `Blocking.inner` -- cottax's own docs say so: *"nothing builds
a nested one yet ... `schedule_for` still refuses MDF"*. That is an upstream feature, not
a workaround this repo can apply.

SAND is the standard alternative, and cottax's `problem.py` was built for it:
`Optimise + RootFind -> Optimise` puts the coupling unknowns into `design` and their
residuals into `equalities` (`~/jaxgraph/src/cottax/problem.py:104-110`). One driver, one
Jacobian, no nested convergence tolerance -- at the cost of a larger design vector, and
of every coupling variable needing a starting guess and a scale that PROCESS never had to
supply because it never exposed them as unknowns.

**Not the whole graph as one opaque block.** Only the nodes genuinely between the design
variables and the conditions end up inside the `Drive`; the acyclic remainder stays
ordinary `Call` steps that run before and after it. `sand_shape()` reports the split.

What is faithful to PROCESS, and what is not
--------------------------------------------
- The **normalised residual** (index 1 of `constraints.leq`/`geq`/`eq`'s 4-tuple) is the
  condition, never the raw residual: PROCESS hands VMCON `-normalised_residual`
  (`process/core/solver/constraints.py:2007`) precisely because the raw one spans 1e-10
  to 9e7 on this run. `VmconDriver` applies the sign flip; see its docstring.
- The **equality/inequality split is positional and user-chosen**, not a property of the
  constraint function: `numerics.icc[:n_equality_constraints]` are the equalities.
  Constraint 16's ported body is a `geq` and it is an *equality* in
  `stellarator_helias.IN.DAT` (which sets `n_equality_constraints = 2` at line 12).
  `constraint_nodes` therefore takes `n_equality` and never inspects the body.
- The **objective's sign** is folded into the objective node's `fn` at assembly time
  (`np.sign(i_figure_merit)`, PROCESS's own `objectives.py:54,105`), because `Optimise`'s
  contract is minimise and a driver that silently maximises is a driver whose `drives`
  claim is a lie.
- **Bounds and scaling are the driver's**, not the problem's -- `Optimise` has nowhere to
  put them and VMCON takes them as separate arguments. `design_bounds` reads PROCESS's
  own per-`ITERATION_VARIABLES` defaults; a caller with a real `IN.DAT`'s
  `numerics.boundl`/`boundu` overrides should pass those instead.

Failing loudly on a constraint that cannot be assembled
-------------------------------------------------------
`constraint_nodes` raises on any active `icc` entry whose arguments do not all resolve.
This is deliberate and it is the point: **an `Optimise` over 12 of PROCESS's 14 active
constraints is a different problem**, and comparing its answer to PROCESS's converged
point would be meaningless. A caller that genuinely wants the reduced problem must ask
for it by name (`omit=`), which puts the omission in the result rather than in a
hardcoded list nobody re-reads.
"""

import dataclasses
import functools
import inspect

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from cottax.blocking import Blocking
from cottax.evaluate import Drive, Schedule, schedule_for
from cottax.graph import Graph
from cottax.plan import Insert, Plan
from cottax.problem import Driven, FixedPoint, Optimise, conditions_of
from cottax.rewrites import Assign, Combine, Residualise

from functional_process.core.solver.drivers import VmconDriver
from cottax.spec import CallableNode, In, NodePath, Out, VarPath
from cottax.tools.minting import MintKey, prefix_path
from cottax.tools.path import path_map
from jax.tree_util import GetAttrKey

from functional_process.core.solver import constraints as ported_constraints
from functional_process.core.solver import objectives as ported_objectives
from functional_process.mda import assign_drivers, cut_graph, default_drivers
from process.core.model import DataStructure
from process.core.solver.iteration_variables import ITERATION_VARIABLES
from process.data_structure.numerics import FiguresOfMerit

COND = MintKey("cond")
"""The namespace constraint/objective values are minted into -- the same one
`Compare`/`Residualise` open. `~/jaxgraph`'s own `CLAUDE.md` explains why it is `^cond`
and not `^res`: *"what a `Problem` reads is a condition whatever shape it has, which with
`Optimise` in the picture covers an objective as well"*."""

REFERENCE_SWITCH_VALUES = {
    "i_rad_loss": 1,  # `.physics.i_rad_loss`
    # `.physics.i_plasma_ignited`, `stellarator_helias.IN.DAT:126`
    "i_plasma_ignited": 1,
    "i_beta_component": 0,  # `.physics.i_beta_component`
    "istell": 6,  # `.stellarator.istell`, `stellarator_helias.IN.DAT:137`
}
"""Static switch arguments of the reference run's active constraints, and their values.

A constraint parameter is bound as a **static** `functools.partial` keyword if and only
if its name is a key here; everything else is resolved to a `VarPath` and becomes an
`In`. That is a deliberately mechanical rule with two checks on it rather than a
judgement made per constraint:

- `test_sand.py::test_switch_values_match_the_contracts_static_argnames` asserts, for
  every active constraint, that this set intersected with the signature is *exactly* the
  `static_argnames` its `Tier1Contract` in `core/solver/test_constraints.py` declares --
  so neither a switch can leak onto a port (it would be traced into an `if`) nor a real
  variable be frozen to a constant.
- `sand_harness.py`'s `switch_audit` equivalent checks each value against the converged
  run itself, the same discipline `mda_harness.switch_audit` applies to node kwargs after
  five registration bugs of exactly this shape.

Values are per *run*, so a caller modelling a different `IN.DAT` passes its own dict --
`switch_values_for` builds one from the run's own initialised `DataStructure`."""


SWITCH_PARAMETER_NAMES = (
    "bkt_life_csf",
    "i_beta_component",
    "i_cp_lifetime",
    "i_density_limit",
    "i_plant_availability",
    "i_plasma_ignited",
    "i_q95_fixed",
    "i_rad_loss",
    "i_tf_bucking",
    "i_tf_inside_cs",
    "i_tf_sup",
    "ibkt_life",
    "ireactor",
    "istell",
    "itart",
)
"""Every parameter name that is a **switch** anywhere in the ported constraint/objective
surface -- the union of the `static_argnames` the `Tier1Contract`s in
`tests/functional_process/core/solver/test_constraints.py` and `test_objectives.py`
declare. A name's *presence here* is what makes `_bind` freeze it
(`REFERENCE_SWITCH_VALUES`' own docstring states the rule); the per-run *value* comes
from the run (`switch_values_for`). Curated rather than parsed out of the test files at
runtime, because the port modules must not import their own tests; the tokamak analogue
of `test_sand.py::test_switch_values_match_the_contracts_static_argnames` is what keeps
the two from drifting."""


def switch_values_for(data, icc, i_figure_merit):
    """Static switch arguments for one run, read off its **initialised**
    `DataStructure` -- `init_process`'s answer to the file plus PROCESS's own defaults,
    so no default is transcribed here.

    The names are the active constraints' and the objective's parameters intersected
    with `SWITCH_PARAMETER_NAMES`; each is resolved to the single `DataStructure` area
    holding a field of that name (the same uniqueness rule `_Resolver` applies, for the
    same reason) and read as an `int`, because a switch selects a formula and is never
    a trace-time array.

    `data` should be a *cold* (initialised, un-run) structure -- `render_xdsm.
    cold_reference(...).data` is the canonical source -- though nothing here could tell
    the difference: no PROCESS switch is an iteration or scan variable
    (`machine_from_indat`'s docstring carries the grep), so a converged structure holds
    the same values.

    `REFERENCE_SWITCH_VALUES` stays as the stellarator reference run's hand-audited
    dict (and the default everywhere, unchanged); this is the mechanical route for any
    other `IN.DAT`.

    Raises
    ------
    ValueError
        On a switch name that resolves to no `DataStructure` area, or to more than
        one -- resolution is by unique field name, the same rule (and the same
        failure mode) as `_Resolver`'s second stage.
    """
    needed = set()
    for cid in icc:
        fn = getattr(ported_constraints, f"constraint_{cid}", None)
        if fn is None:
            # `constraint_nodes` raises on this, with the message that names the id;
            # a second, earlier copy of the refusal here would just shadow it.
            continue
        needed |= set(inspect.signature(fn).parameters) & set(SWITCH_PARAMETER_NAMES)
    merit = FiguresOfMerit(abs(int(i_figure_merit)))
    metric = ported_objectives.OBJECTIVE_METRICS[merit]
    needed |= set(inspect.signature(metric).parameters) & set(SWITCH_PARAMETER_NAMES)
    areas = _data_structure_areas()
    values = {}
    for name in sorted(needed):
        hits = [a for a in areas if hasattr(getattr(data, a), name)]
        if len(hits) != 1:
            raise ValueError(
                f"switch {name!r} is "
                + (
                    "in no `DataStructure` area"
                    if not hits
                    else f"ambiguous across `DataStructure` areas {hits}"
                )
                + " -- it cannot be read mechanically, pass `switch_values` by hand"
            )
        values[name] = int(getattr(getattr(data, hits[0]), name))
    return values


def iteration_variable_path(ixc_id: int) -> VarPath:
    """The `VarPath` of PROCESS iteration variable `ixc_id`.

    `ITERATION_VARIABLES[id]`'s `(module, target_name or name)` pair *is* cottax's
    two-key path: `load_iteration_variables` does `getattr(data, iv.module)` and
    `set_scaled_iteration_variable` does `getattr(module, iv.target_name or iv.name)`
    (`process/core/solver/iteration_variables.py:290,354`), so this is PROCESS's own
    accessor spelled structurally. `target_name` is the storage location and `name` the
    human label -- cottax already separates those, so only the former is a path.

    Raises
    ------
    NotImplementedError
        If the variable addresses one element of an array field (`array_index`, IDs
        125-136). A `SequenceKey` component resolves for a *read*, but `Graph`
        refuses an output that lies inside a variable another node reads whole, and
        every consumer of impurity fractions reads the whole array. See
        `_audit/optimise_design.md` §1.3 -- this needs per-element `VarPath`s at the
        port level, not a key.
    """
    iteration_variable = ITERATION_VARIABLES[ixc_id]
    if iteration_variable.array_index is not None:
        raise NotImplementedError(
            f"iteration variable {ixc_id} ({iteration_variable.name}) addresses element "
            f"{iteration_variable.array_index} of "
            f"{iteration_variable.target_name or iteration_variable.name} -- an "
            f"`Optimise` cannot own an array element while any node reads the enclosing "
            f"array, which `Graph.__check_init__` refuses outright. See "
            f"`_audit/optimise_design.md` §1.3"
        )
    return VarPath((
        GetAttrKey(iteration_variable.module),
        GetAttrKey(iteration_variable.target_name or iteration_variable.name),
    ))


def design_bounds(ixc):
    """`((VarPath, lower, upper), ...)` from `ITERATION_VARIABLES`' own defaults, ready
    for `VmconDriver.bounds`.

    These are PROCESS's *table* bounds (`initialise_iteration_variables`,
    `iteration_variables.py:414-417`). A real run may override them per input file
    through `numerics.boundl`/`boundu`; a caller that has a converged `DataStructure` in
    hand should build the tuple from those instead, which is what `sand_harness.py` does.
    """
    return tuple(
        (
            iteration_variable_path(i),
            float(ITERATION_VARIABLES[i].lower_bound),
            float(ITERATION_VARIABLES[i].upper_bound),
        )
        for i in ixc
    )


def _data_structure_areas():
    return [f.name for f in dataclasses.fields(DataStructure())]


class _Resolver:
    """`parameter name -> VarPath`, graph first, `DataStructure` second.

    Two stages, and the order matters. A name that some node in `graph` already owns or
    reads resolves to *that* `VarPath`, so a constraint node is wired into the real
    dataflow rather than to a same-named field in another area. Only a name the graph has
    never heard of falls back to scanning `DataStructure`'s areas -- which is the correct
    answer for a constraint's *bound* (`pflux_fw_neutron_max_mw`,
    `sig_tf_wp_max`, ...): those are user inputs, and reading one simply adds a boundary
    input.

    Both stages require the match to be **unique** and raise naming the candidates
    otherwise. PROCESS's own naming convention is what makes this work (field names are
    globally unique across areas in practice, `documentation/source/development/
    standards.md`), and it is checked rather than assumed -- the same rule
    `mda_constraint_harness._resolve_args` and `mda_harness._backing_field` already use.
    """

    def __init__(self, graph: Graph):
        self.by_name = {}
        for var in graph.variables:
            keys = var.keys
            if len(keys) == 2 and all(isinstance(k, GetAttrKey) for k in keys):
                self.by_name.setdefault(keys[-1].name, set()).add(var)
        self.areas = _data_structure_areas()
        self.data = DataStructure()

    def __call__(self, name: str) -> VarPath:
        hits = self.by_name.get(name)
        if hits and len(hits) == 1:
            return next(iter(hits))
        if hits:
            raise ValueError(
                f"{name!r} names {len(hits)} variables in the graph "
                f"({sorted(v.path_str() for v in hits)}) -- resolution is by unique "
                f"name, so this one has to be given explicitly"
            )
        areas = [a for a in self.areas if hasattr(getattr(self.data, a), name)]
        if len(areas) == 1:
            return VarPath((GetAttrKey(areas[0]), GetAttrKey(name)))
        raise ValueError(
            f"{name!r} is in no node of the graph and "
            + (
                "no `DataStructure` area has a field of that name"
                if not areas
                else f"is ambiguous across `DataStructure` areas {areas}"
            )
        )


def _bind(fn, resolve, switch_values):
    """`(CallableNode-ready fn, inputs)` for one ported constraint/objective function.

    Static switch arguments are bound with `functools.partial` at assembly time -- they
    select a formula, carry no derivative and take part in no edge
    (`_audit/naming_convention.md` § "switches are not ports"). Everything else becomes
    one positional `In`.
    """
    parameters = list(inspect.signature(fn).parameters)
    static = {p: switch_values[p] for p in parameters if p in switch_values}
    read = [p for p in parameters if p not in static]
    bound = functools.partial(fn, **static) if static else fn
    return bound, read, tuple(In(resolve(p)) for p in read)


def constraint_nodes(graph, icc, n_equality, switch_values=None, omit=()):
    """One `CallableNode` per active constraint, plus the equality/inequality split.

    Parameters
    ----------
    graph :
        The graph the constraints will be inserted into -- used only to resolve
        parameter names to `VarPath`s (see `_Resolver`).
    icc :
        `numerics.icc`, PROCESS's active constraint IDs **in order**. The order is
        load-bearing: the split below is positional.
    n_equality :
        `numerics.n_equality_constraints`. The first `n_equality` entries of `icc` are
        the equalities, whatever their ported body's `leq`/`geq`/`eq` says --
        constraint 16 is the standing counterexample (a `geq` body driven to equality by
        `stellarator_helias.IN.DAT:12`).
    switch_values :
        Static switch arguments, defaulting to `REFERENCE_SWITCH_VALUES`.
    omit :
        Constraint IDs to leave out **deliberately**. Anything omitted is returned so a
        caller can report it; nothing is dropped silently.

    Returns
    -------
    :
        `(nodes, equalities, inequalities, omitted)` -- `nodes` a `{NodePath:
        CallableNode}` dict, `equalities`/`inequalities` tuples of the `^cond.*`
        `VarPath`s in `icc` order, `omitted` a `{id: reason}` dict.

    Raises
    ------
    ValueError
        On an active constraint whose arguments do not all resolve and that is not in
        `omit`. See this module's docstring for why this is not a warning.
    """
    switch_values = REFERENCE_SWITCH_VALUES if switch_values is None else switch_values
    resolve = _Resolver(graph)
    nodes, equalities, inequalities, omitted = {}, [], [], {}
    for position, cid in enumerate(icc):
        if cid in omit:
            omitted[cid] = "omitted by the caller"
            continue
        fn = getattr(ported_constraints, f"constraint_{cid}", None)
        if fn is None:
            raise ValueError(
                f"constraint {cid} is active in this run but `core/solver/"
                f"constraints.py` has no `constraint_{cid}`"
            )
        try:
            bound, read, inputs = _bind(fn, resolve, switch_values)
        except ValueError as e:
            raise ValueError(
                f"constraint {cid} cannot be assembled: {e}. An `Optimise` missing one "
                f"of PROCESS's active constraints solves a different problem -- pass "
                f"`omit={{{cid}}}` to leave it out on purpose and have it reported"
            ) from e
        condition = prefix_path(
            VarPath((GetAttrKey("constraints"), GetAttrKey(f"c{cid}"))), COND
        )
        nodes[NodePath((GetAttrKey(f"Constraint{cid}"),))] = CallableNode(
            inputs=inputs,
            outputs=(Out(condition),),
            # index 1 of `(residual, normalised_residual, value, bound)` -- see the
            # module docstring.
            fn=_NormalisedResidual(bound, tuple(read)),
        )
        (equalities if position < n_equality else inequalities).append(condition)
    return nodes, tuple(equalities), tuple(inequalities), omitted


class _NormalisedResidual(eqx.Module):
    """`fn(*args) -> normalised_residual`, as an `eqx.Module` rather than a closure.

    A node definition is a jit cache key, so a body rebuilt per access must still compare
    equal to the last one -- exactly the reason `cottax.rewrites.Compare` uses a
    `Pairwise` module instead of a lambda.
    """

    fn: object
    names: tuple

    def __call__(self, *args):
        return self.fn(**dict(zip(self.names, args, strict=True)))[1]


class _SignedMetric(eqx.Module):
    """`sign * objective_metric(*args)`. Same not-a-closure reasoning as
    `_NormalisedResidual`.
    """

    fn: object
    sign: float

    def __call__(self, *args):
        return self.sign * self.fn(*args)


def objective_node(graph, i_figure_merit, switch_values=None):
    """One `CallableNode` computing the run's figure of merit, and the `VarPath` it owns.

    `_audit/next_steps.md` §6 and `CLAUDE.md` both say the objective is "a query, not a
    node". That is right about *which* -- `Optimise.objective` is a single `In`, i.e. one
    `VarPath` -- and wrong about the metric: fourteen of the sixteen
    `objective_metric_<id>` functions are arithmetic (`0.2 * rmajor`, `coe / 100`), and
    arithmetic needs a body. The node is still per-query: it does not exist until an
    `Optimise` is assembled, and a different `i_figure_merit` mints a different node.

    The sign is folded in here: PROCESS applies `np.sign(i_figure_merit)` outside the
    branch (`process/core/solver/objectives.py:54,105`), negative meaning maximise, and
    `Optimise`'s contract is minimise.
    """
    switch_values = REFERENCE_SWITCH_VALUES if switch_values is None else switch_values
    merit = FiguresOfMerit(abs(int(i_figure_merit)))
    fn = ported_objectives.OBJECTIVE_METRICS[merit]
    resolve = _Resolver(graph)
    parameters = list(inspect.signature(fn).parameters)
    static = {p: switch_values[p] for p in parameters if p in switch_values}
    read = [p for p in parameters if p not in static]
    bound = functools.partial(fn, **static) if static else fn
    objective = prefix_path(VarPath((GetAttrKey("numerics"), GetAttrKey("objf"))), COND)
    node = CallableNode(
        inputs=tuple(In(resolve(p)) for p in read),
        outputs=(Out(objective),),
        fn=_SignedMetric(bound, float(np.sign(i_figure_merit))),
    )
    return NodePath((GetAttrKey("Objective"),)), node, objective


def optimise_graph(
    graph,
    ixc,
    icc,
    n_equality,
    i_figure_merit,
    driver=None,
    switch_values=None,
    omit=(),
):
    """`graph` with the constraint nodes, the objective node and one `Optimise` inserted.

    The `Optimise` owns the active iteration variables' `VarPath`s. On the reference run
    all eight are **boundary inputs** of the graph, so nothing collides: registering the
    problem turns eight free inputs into eight owned variables and changes nothing else.
    An `ixc` whose variable *is* produced by a node (7 of the 83-entry table are, e.g.
    ID 1 `.physics.aspect`) is a duplicate-ownership conflict `Graph.__check_init__`
    refuses loudly -- the policy `total_process.py`'s `DefaultAspectRatio` already states
    for ID 1 (drop the producer when the ixc is active) generalises, but is not applied
    here because no such ID is active in this run.

    Returns
    -------
    :
        `(graph, problem_name, report)` -- `report` a dict with the design/equality/
        inequality `VarPath`s and any `omitted` constraints.
    """
    design = tuple(iteration_variable_path(i) for i in ixc)
    nodes, equalities, inequalities, omitted = constraint_nodes(
        graph, icc, n_equality, switch_values, omit
    )
    objective_name, objective_definition, objective = objective_node(
        graph, i_figure_merit, switch_values
    )
    nodes[objective_name] = objective_definition
    problem_name = NodePath((GetAttrKey("Opt"),))
    nodes[problem_name] = Optimise(
        objective=In(objective),
        design=tuple(Out(v) for v in design),
        equalities=tuple(In(c) for c in equalities),
        inequalities=tuple(In(c) for c in inequalities),
    )
    inserted = (Plan(graph) + Insert(path_map(nodes.items()))).graph
    # **No driver is attached here by default, and that is the ordering the new API
    # forces.** `Combine` refuses to join two problems that carry an algorithm -- *"one
    # discards the algorithm answering each, `Undrive` first"* -- and this graph's whole
    # purpose is to join every `FixedPoint` into one `Optimise`. So SAND builds on
    # `mda.cut_graph` (structure, no drivers), joins, and assigns afterwards. A `driver`
    # may still be passed for a caller that wants one attached immediately; it carries
    # the caller's data (`bounds`, `condition_scale`, `callback`), none of which has a
    # home on `Optimise` and never did.
    if driver is not None:
        inserted = Assign(problem_name, driver).apply(inserted)
    return (
        inserted,
        problem_name,
        {
            "design": design,
            "equalities": equalities,
            "inequalities": inequalities,
            "objective": objective,
            "omitted": omitted,
        },
    )


def degenerate_fixed_points(graph, env, problems=None):
    """`FixedPoint` problems whose residual `g(u) - u` is *structurally* zero here.

    A `FixedPointFunction` that is an identity in the active configuration is a
    perfectly well-posed Picard problem -- it converges in one step from anywhere -- and
    a **rank-deficient SAND equality**: `r = g(u) - u == 0` for every `u` determines
    nothing, so the equality block is singular and any SQP fails on it. Two of this
    graph's fixed points *were* in that state under the reference configuration
    (`EtaTurbineStep` at `i_thermal_electric_conversion == 2` and `CplifeAvail` at
    `itart == 0`) -- **and neither exists any more**. Splitting the switch each carried
    showed the identity arm is PROCESS saying the field is an *input*, so the tree spells
    it as an empty slot instead of a residual to differentiate
    (`_audit/next_steps.md` §14.2/§14.11). This function stays, and its shape is the
    reason: it recovers the property at runtime for *any* configuration, where the tree
    states it only for the switches someone has split.

    Detected, not listed: each candidate's `d(g(u) - u)/du` is differentiated at `env`'s
    own values and reported degenerate when the whole row **and** column vanish. Listing
    them by name would bake one configuration into the code, which is exactly what
    `machine_from_indat` exists to avoid.

    Returns
    -------
    :
        `tuple[NodePath, ...]` -- the degenerate problems, in binding order.
    """
    if problems is None:
        problems = tuple(n for n in graph.declared if isinstance(graph[n], FixedPoint))
    degenerate = []
    for problem in problems:
        definition = graph[problem]
        # `conditions_of`, not `.reads`: a problem that has been through `Initialise`
        # also reads its `Start` port(s), and those are driver data, not conditions.
        # Including them put a `^guess.*` in `step`'s output stack, where `env` has no
        # value for it -- a `KeyError` the bare `except` below then swallowed, so every
        # fixed point silently reported "not degenerate" and the two identity ones
        # (`eta_turbine_step`, `cplife_avail`, both since deleted by the switch
        # conversion) reached `reduce_jacobian` as exactly-zero rows of `J_RY`, i.e. a
        # singular equality block.
        owns, reads = definition.owns, conditions_of(definition)
        producers = {r: graph.owners[r] for r in reads if r in graph.owners}
        body = graph.subgraph(tuple(set(producers.values())))

        def step(*unknowns, _body=body, _owns=owns, _reads=reads):
            values = dict(env)
            values.update(zip(_owns, unknowns, strict=True))
            from cottax.evaluate import _run_acyclic

            out = _run_acyclic(_body, values)
            return jnp.stack([jnp.asarray(out[r]) for r in _reads])

        try:
            start = tuple(jnp.asarray(env[v]) for v in owns)
            jacobian = jax.jacfwd(step, argnums=tuple(range(len(owns))))(*start)
            block = np.stack(
                [np.asarray(c, dtype=float).reshape(len(reads)) for c in jacobian],
                axis=1,
            ) - np.eye(len(owns))
        except Exception:  # noqa: BLE001 -- undetectable is not degenerate
            continue
        if np.allclose(block, 0.0):
            degenerate.append(problem)
    return tuple(degenerate)


def array_valued_problems(graph, env, problems=None):
    """Declared problems owning a **non-scalar** unknown at `env`'s own values -- the
    ones today's SAND layer cannot absorb, detected rather than listed.

    The design side of `VmconDriver` flattens pytrees (`ravel_pytree`), but everything
    per-*condition* is scalar arithmetic: `scaled_problem` stacks conditions with
    `jnp.stack` (one scalar each), keys `condition_scale` by condition, sizes the
    bound arrays by `len(unknowns)`, and splits equalities from inequalities by
    *condition count*. `residual_condition_scales`' `1/|u|` is a scalar per unknown for
    the same reason. An array-unknown `FixedPoint` residualised into the combined
    `Optimise` therefore fails at the first of those seams (`float()` of a matrix), and
    making the layer element-wise is a driver extension with its own validation -- a
    separate, recorded decision, the same standing as `_audit/optimise_design.md` §1.3
    refusing array-element iteration variables.

    The standing instance is the tokamak PF-coil ring, cut at
    `.pf_coil.ind_pf_cs_plasma_mutual` (a circuit-by-circuit mutual-inductance matrix)
    and `.pf_coil.n_pf_coil_turns` (a per-coil vector) -- `mda.CUTS`' own docstring for
    the cycle. A caller that deletes such a problem freezes its loop-carried unknowns
    at whatever the env seeds them to (self-consistent when the seed is a converged
    MDA), exactly as deleting a degenerate fixed point does -- and must say so, which
    is why this returns the problems rather than deleting anything itself.

    An unknown with no value in `env` is treated as scalar: nothing could be measured,
    and refusing to guess is `degenerate_fixed_points`' discipline too.
    """
    if problems is None:
        problems = tuple(n for n in graph.declared if isinstance(graph[n], FixedPoint))
    return tuple(
        problem
        for problem in problems
        if any(
            unknown in env and jnp.ndim(jnp.asarray(env[unknown])) > 0
            for unknown in graph[problem].owns
        )
    )


def sand_graph(graph, skip=()):
    """`graph` with every `FixedPoint` (bar `skip`) residualised and every problem
    combined into one `^problem.sand`.

    `Residualise` converts `FixedPoint -> RootFind` (`r = g(u) - u`, always available);
    `Combine` folds `problem.py`'s `+`, and `Optimise + RootFind -> Optimise` **is**
    SAND. Neither `Combine` nor `rewrites.py` knows that -- the join is `problem.py`'s.

    A skipped fixed point keeps its `FixedPoint` problem, which `Combine` then refuses
    (`Optimise` with `FixedPoint` is a `TypeError` naming `Residualise`), so `skip` is
    for problems a caller has *deleted*, not merely left alone. `sand_schedule` handles
    `degenerate_fixed_points` by dropping the problem node outright, which is the
    structurally honest answer: its unknown reverts to an ordinary boundary input.
    """
    plan = Plan(graph)
    residualised = []
    for problem in graph.declared:
        if problem in skip or not isinstance(graph[problem], FixedPoint):
            continue
        plan = plan + Residualise(problem)
        residualised.append(problem)
    plan = plan + Combine(NodePath((GetAttrKey("sand"),)), tuple(plan.graph.declared))
    return plan.graph, tuple(residualised)


def constraints_outside_block(graph):
    """Active constraints whose node falls **outside** the combined problem's own SCC
    block -- `{constraint id: NodePath}` -- which today's evaluation seam cannot carry.

    A constraint node joins the problem's block exactly when it reads something the
    block produces (the problem reads its `^cond.*` back, closing the cycle). One
    whose every input is a boundary value -- or produced only by nodes upstream of
    every problem unknown -- is a plain `Call` step instead, and its `^cond.*` then
    reaches **nobody**: `Drive.context` is the *body's* inputs, the body is
    `subgraph.runnable` (the problem taken out), and only the problem reads a
    condition, so a condition produced outside the block is in neither the body's
    outputs nor the map's context and `ConditionMap.__call__` dies on a `KeyError`.
    That is arguably an upstream seam (a condition constant over the block's unknowns),
    but it is also a true statement about the *problem*: such a constraint is not a
    function of anything the SQP moves, in this graph, and pretending otherwise would
    hand VMCON a permanently-fixed row with an all-zero gradient.

    The standing instances are `large_tokamak_eval.IN.DAT`'s TF/CS stress and
    superconductor-margin constraints, every one of whose value reads is a boundary
    input because the producing PROCESS models (`superconducting.py`'s conductor
    performance and stress blocks, `pfcoil.py`'s CS criticals) are unported -- the
    missing-producer audit in the harness report names each. The honest assembly
    *omits* them (`optimise_graph(omit=...)`, so they are reported, never dropped
    silently) and states their values at the seed separately.

    Empty on the stellarator reference run -- every one of its 14 active constraints
    reads at least one block-produced variable -- so nothing changes there.
    """
    blocking = Blocking.scc(graph)
    problem_block = next(
        frozenset(nodes)
        for nodes, problem in zip(blocking.blocks, blocking.problems, strict=True)
        if problem is not None
    )
    outside = {}
    for name in graph.nodes:
        leaf = name.keys[-1].name
        if leaf.startswith("Constraint") and name not in problem_block:
            outside[int(leaf.removeprefix("Constraint"))] = name
    return outside


def residual_condition_scales(drive, env, floor=1e-12):
    """`((condition, factor), ...)` for exactly the SAND residual conditions, ready for
    `VmconDriver.condition_scale`.

    A residual `r = g(u) - u` carries `u`'s units, so its natural scale is `1/|u|` --
    which turns it into a *relative* residual, the same O(1) shape PROCESS's own
    `normalised_residual` already has. Every real constraint and the objective keep a
    factor of `1.0`, so this changes nothing PROCESS would recognise; see
    `VmconDriver.condition_scale` for the measurement that made it necessary.

    The pairing is by **place, not position**: `Residualise` mints `^cond.<condition>`
    from a `FixedPoint` whose own condition is already `^cond.<u>`, so the residual for
    unknown `.X` is `^cond^cond.X`; a `FixedPointCut`'s is `^cond.X` against the unknown
    `^hat.X`; and an `ImplicitFunction`'s (`Intersect`) is `^cond.X` against `.X`.
    Stripping every minted root with `unminted` collapses all three to the same place,
    which is what makes one rule cover them.

    An unknown with no scale of its own keeps a factor of `1.0`
    -------------------------------------------------------------
    `floor` is the magnitude below which `|u|` stops being a usable scale, and the
    answer there is **`1.0`, not `1/floor`**. Clamping the *magnitude* instead --
    `1 / max(|u|, floor)` -- looks like the same guard and is not: it turns "this
    quantity has no natural scale" into "weight this row by `1e12`", which is the
    largest number in the whole problem by nine orders of magnitude.

    **Measured, and this is the bug it caused.** `.power.qac` (cryogenic AC-loss load,
    one of the four the cryo `q*` node owns -- `CryoQLoadsStep`'s unknowns then, an
    ordinary `CryoQLoads` occupant's outputs now) is *identically* zero on the reference
    run
    -- `ensxpfm = 0`, there being no PF energy swing to dissipate -- so its residual row
    is exactly `(0, ..., -1, ..., 0)`: the trivial, perfectly well-posed equality
    `u = 0`. Clamped, that row entered VMCON's QP at `-1e12`, nine orders of magnitude
    above every other row, and the Jacobian VMCON actually sees (rows by their condition
    scale, columns by `VmconDriver.scaled`'s `1/x_start`) had condition number
    **6.68e12**, equality block `2.00e12`. With the row left alone: **2.09e4** and
    `6.25e3`. Stage C2 from PROCESS's own converged point ran to `max_iter = 100`
    without formally converging, oscillating around `objf ~ 1.2179`.

    Two honest qualifications on that last sentence, both measured. The `max_iter`
    symptom was recorded on one tree state; on a later one, an A/B of the two rules in a
    single process off one PROCESS run gives **73** iterations clamped against **62**
    here, both converging -- so the clamp is not always fatal, and iteration count on
    this problem is noisy enough (33 to 73 across a sweep of factors for this one row)
    that it is weak evidence on its own. The conditioning figure is the durable one. And
    a fifth to a third of the QP subproblems solve inaccurately (`cvxpy`'s own warning:
    26 of 73 clamped, 21 of 62 here) either way, so this row was never the *only* thing
    straining the QP -- see `VmconDriver.condition_scale` for what is left.

    `1.0` is also the value the *design* side already degrades to for exactly the same
    reason -- `VmconDriver.scaled` keeps `scale = 1.0` for an unknown starting at `0.0`
    (its docstring says so) -- so the two scalings now agree instead of one degrading
    to neutral while the other blows up. Neutral is the honest reading: a residual
    whose unknown is zero has no relative form, and its row is already O(1) in the
    only column it touches.
    """
    from cottax.tools.minting import unminted

    def place(path):
        while (stripped := unminted(path)) != path:
            path = stripped
        return path

    unknowns = {place(v): v for v in drive.unknowns}
    scales = []
    for condition in drive.conditions:
        if condition.path_str().startswith(("^cond.constraints.", "^cond.numerics.")):
            continue
        unknown = unknowns.get(place(condition))
        if unknown is None or unknown not in env:
            continue
        magnitude = abs(float(np.asarray(env[unknown])))
        usable = np.isfinite(magnitude) and magnitude > floor
        scales.append((condition, 1.0 / magnitude if usable else 1.0))
    return tuple(scales)


def sand_schedule(
    graph,
    problem_name,
    driver=None,
    bounds=(),
    callback=None,
    condition_scale=(),
    max_iter=None,
):
    """A `Schedule` for `graph`'s single `^problem.sand`, answered by `driver`.

    `driver` defaults to a `VmconDriver` whose equality/inequality counts are read off
    the combined `Optimise` **node definition**, not counted by the caller -- the
    positional contract `_audit/optimise_design.md` §4.1 warns about only exists if
    somebody chooses to count.

    `max_iter` is forwarded the same way, and `None` keeps the driver's own default --
    see `mda.default_drivers` for what that default is and why a SAND block outgrows it.
    """
    blocking = Blocking.scc(graph)
    drivers = default_drivers(
        graph,
        bounds=bounds,
        callback=callback,
        condition_scale=condition_scale,
        max_iter=max_iter,
    )
    if driver is not None:
        problem = next(
            p
            for p, t in zip(blocking.problems, blocking.problem_types, strict=True)
            if t is not None and issubclass(t, Optimise)
        )
        drivers[problem] = driver
    # Drivers go into the graph (`Assign`), and `schedule_for` reads them from there.
    return schedule_for(Blocking.scc(assign_drivers(blocking.graph, drivers)))


def sand_shape(schedule: Schedule) -> dict:
    """The one `Drive`'s size, for reporting: how much of the graph is actually inside
    the solved block and how much still runs as ordinary `Call` steps.
    """
    drive = next(step for step in schedule.steps if isinstance(step, Drive))
    node = drive.subgraph[drive.problem]
    # `Driven` forwards `inputs`/`outputs` and nothing else -- the problem-specific
    # properties are reached through `.problem`, deliberately: *"a driven node **has** a
    # problem, it is not one"*. Spelling it out says the true thing about the shape.
    definition = node.problem if isinstance(node, Driven) else node
    return {
        "drive_nodes": len(drive.nodes),
        "unknowns": len(drive.unknowns),
        "conditions": len(drive.conditions),
        "context": len(drive.context),
        "design": len(definition.design),
        "equalities": len(definition.equalities),
        "inequalities": len(definition.inequalities),
        "schedule_steps": len(schedule.steps),
        "drive": drive,
    }


def reference_problem(
    graph,
    ixc,
    icc,
    n_equality,
    i_figure_merit,
    env,
    switch_values=None,
    omit=(),
):
    """The whole assembly in one call: cut, register, drop degenerate fixed points,
    residualise, combine.

    `env` is used only to detect the degenerate fixed points (`degenerate_fixed_points`);
    it is not a starting guess and nothing here solves anything.

    Returns
    -------
    :
        `(graph, problem_name, report)` -- `report` carries `design`/`equalities`/
        `inequalities`/`omitted` from `optimise_graph`, plus `degenerate` and
        `residualised`.
    """
    from cottax.plan import Delete

    driven = cut_graph(graph)
    degenerate = degenerate_fixed_points(driven, env)
    if degenerate:
        driven = Delete(degenerate).apply(driven)
    with_problem, problem_name, report = optimise_graph(
        driven, ixc, icc, n_equality, i_figure_merit, switch_values, omit
    )
    combined, residualised = sand_graph(with_problem)
    report["degenerate"] = degenerate
    report["residualised"] = residualised
    return combined, problem_name, report
