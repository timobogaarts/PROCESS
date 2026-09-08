"""The `Optimise` layer: PROCESS's own optimisation problem, assembled onto the graph
and solved as **SAND** (Simultaneous ANalysis and Design).
"""

import dataclasses
import functools
import inspect

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from cottax.blocking import Blocking
from cottax.evaluate import Drive, Schedule
from cottax.graph import Graph
from cottax.plan import Insert, Plan
from cottax.problem import Driven, FixedPoint, Optimise, conditions_of, is_fixed_point, is_optimise
from cottax.rewrites import Assign, Combine, NestInside, Residualise

from cottax.spec import ImplementedFunction, In, NodePath, Out, VarPath
from cottax.tools.minting import MintKey, prefix_path
from cottax.tools.path import path_map
from jax.flatten_util import ravel_pytree
from jax.tree_util import GetAttrKey, SequenceKey

from functional_process.cottax.core.solver import constraints as ported_constraints
from functional_process.cottax.core.solver import objectives as ported_objectives
from functional_process.cottax.mda import assign_drivers, cut_graph, default_drivers
from functional_process.vocabulary import (
    AREAS,
    ITERATION_VARIABLES,
    FiguresOfMerit,
)
from functional_process.vocabulary.input_variables import INPUT_VARIABLES

METRIC = MintKey("metric")
"""The namespace a figure of merit is minted into **before** its direction is applied.
"""

COND = MintKey("cond")
"""The namespace constraint/objective values are minted into -- the same one
`Compare`/`Residualise` open.
"""

REFERENCE_SWITCH_VALUES = {
    "i_rad_loss": 1,  # `.physics.i_rad_loss`
    # `.physics.i_plasma_ignited`, `stellarator_helias.IN.DAT:126`
    "i_plasma_ignited": 1,
    "i_beta_component": 0,  # `.physics.i_beta_component`
    "istell": 6,  # `.stellarator.istell`, `stellarator_helias.IN.DAT:137`
}
"""Static switch arguments of the reference run's active constraints, and their values.
"""


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
`functional_process/tests/cottax/core/solver/test_constraints.py` and
`test_objectives.py` declare.
"""


def switch_values_for(data, icc, i_figure_merit):
    """Static switch arguments for one run, read off its **initialised** `DataStructure`
    -- `init_process`'s answer to the file plus PROCESS's own defaults, so no default is
    transcribed here.
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
    areas = AREAS
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
    """The `VarPath` of PROCESS iteration variable `ixc_id`."""
    iteration_variable = ITERATION_VARIABLES[ixc_id]
    keys = (
        GetAttrKey(iteration_variable.module),
        GetAttrKey(iteration_variable.target_name or iteration_variable.name),
    )
    if iteration_variable.array_index is not None:
        keys = (*keys, SequenceKey(iteration_variable.array_index))
    return VarPath(keys)


def design_bounds(ixc):
    """`((VarPath, lower, upper), ...)` from `ITERATION_VARIABLES`' own defaults, ready
    for `VmconDriver.bounds`.
    """
    return tuple(
        (
            iteration_variable_path(i),
            float(ITERATION_VARIABLES[i].lower_bound),
            float(ITERATION_VARIABLES[i].upper_bound),
        )
        for i in ixc
    )


NON_INPUT_FIELDS = {
    "available_radial_space": "build",
    "b_cs_peak_flat_top_end": "pf_coil",
    "b_cs_peak_pulse_start": "pf_coil",
    "b_plasma_total": "physics",
    "b_tf_inboard_peak_with_ripple": "tfcoil",
    "beta_beam": "physics",
    "beta_fast_alpha": "physics",
    "beta_poloidal_eps": "physics",
    "beta_poloidal_vol_avg": "physics",
    "beta_thermal_vol_avg": "physics",
    "beta_toroidal_vol_avg": "physics",
    "big_q_plasma": "current_drive",
    "bktcycles": "costs",
    "c_tf_total": "tfcoil",
    "cdirt": "costs",
    "coe": "costs",
    "concost": "costs",
    "coppera_m2": "rebco",
    "cplife": "costs",
    "dr_fw_outboard": "build",
    "dx_tf_inboard_out_toroidal": "tfcoil",
    "eps": "physics",
    "eta_cd_norm_hcd_primary": "current_drive",
    "f_p_beam_shine_through": "current_drive",
    "f_p_plasma_separatrix_rad": "physics",
    "f_pden_alpha_electron_mw": "physics",
    "f_pden_alpha_ions_mw": "physics",
    "f_t_alpha_energy_confinement": "physics",
    "flu_tf_neutron_fast_peak": "fwbs",
    "fzmin": "reinke",
    "j_cs_critical_flat_top_end": "pf_coil",
    "j_cs_critical_pulse_start": "pf_coil",
    "j_cs_pulse_start": "pf_coil",
    "j_tf_wp": "tfcoil",
    "j_tf_wp_critical": "tfcoil",
    "j_tf_wp_quench_heat_max": "tfcoil",
    "life_blkt_fpy": "fwbs",
    "life_div_fpy": "costs",
    "n_beam_decay_lengths_core": "current_drive",
    "n_charge_plasma_effective_vol_avg": "physics",
    "n_cycle": "cs_fatigue",
    "n_iter_vacuum_pumps": "vacuum",
    "nd_beam_ions": "physics",
    "nd_beam_ions_out": "physics",
    "nd_plasma_electron_line": "physics",
    "nd_plasma_electron_on_axis": "physics",
    "nd_plasma_electrons_max": "physics",
    "nd_plasma_ions_total_vol_avg": "physics",
    "p_cp_resistive_mw": "tfcoil",
    "p_cryo_plant_electric_mw": "heat_transport",
    "p_div_bt_q_aspect_rmajor_mw": "physics",
    "p_fusion_total_mw": "physics",
    "p_hcd_injected_electrons_mw": "current_drive",
    "p_hcd_injected_ions_mw": "current_drive",
    "p_hcd_injected_total_mw": "current_drive",
    "p_l_h_threshold_mw": "physics",
    "p_plant_electric_net_mw": "heat_transport",
    "p_plasma_heating_total_mw": "physics",
    "p_plasma_separatrix_mw": "physics",
    "p_plasma_separatrix_rmajor_mw": "physics",
    "p_tf_leg_resistive_mw": "tfcoil",
    "pden_alpha_total_mw": "physics",
    "pden_electron_transport_loss_mw": "physics",
    "pden_ion_electron_equilibration_mw": "physics",
    "pden_ion_transport_loss_mw": "physics",
    "pden_non_alpha_charged_mw": "physics",
    "pden_plasma_core_rad_mw": "physics",
    "pden_plasma_ohmic_mw": "physics",
    "pden_plasma_rad_mw": "physics",
    "peakpoloidalpower": "pf_power",
    "pflux_fw_neutron_mw": "physics",
    "pflux_fw_rad_max_mw": "constraints",
    "plasma_current": "physics",
    "powerht_constraint": "stellarator",
    "powerscaling_constraint": "stellarator",
    "psolradmw": "physics",
    "ptfnucpm3": "fwbs",
    "q95_min": "physics",
    "radius_beam_tangency": "current_drive",
    "radius_beam_tangency_max": "current_drive",
    "rbld": "build",
    "required_radial_space": "build",
    "rminor": "physics",
    "sig_tf_case": "tfcoil",
    "sig_tf_cs_bucked": "tfcoil",
    "sig_tf_wp": "tfcoil",
    "srcktpm": "pf_power",
    "str_wp": "tfcoil",
    "stress_shear_cs_peak": "pf_coil",
    "t_current_ramp_up_min": "constraints",
    "t_plant_pulse_total": "times",
    "tcpav2": "tfcoil",
    "temp_cp_peak": "tfcoil",
    "temp_croco_quench": "tfcoil",
    "temp_cs_superconductor_margin": "pf_coil",
    "temp_fw_peak": "fwbs",
    "temp_plasma_electron_density_weighted_kev": "physics",
    "temp_plasma_ion_density_weighted_kev": "physics",
    "temp_tf_superconductor_margin": "tfcoil",
    "tfcmw": "tfcoil",
    "toroidalgap": "tfcoil",
    "v_tf_coil_dump_quench_kv": "tfcoil",
    "vol_plasma": "physics",
    "vs_cs_pf_total_pulse": "pf_coil",
    "vs_cs_pf_total_ramp": "pf_coil",
    "vs_plasma_ramp_required": "physics",
    "vs_plasma_total_required": "physics",
    "vv_stress_quench": "superconducting_tfcoil",
}
"""`field name -> DataStructure area` for every name the ported constraint/objective
layer can ask for that is **not** a declared PROCESS input.
"""


class _Resolver:
    """`parameter name -> VarPath`, graph first, PROCESS's declared inputs second."""

    def __init__(self, graph: Graph):
        self.by_name = {}
        for var in graph.variables:
            keys = var.keys
            if len(keys) == 2 and all(isinstance(k, GetAttrKey) for k in keys):
                self.by_name.setdefault(keys[-1].name, set()).add(var)

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
        area = NON_INPUT_FIELDS.get(name)
        if area is None:
            declaration = INPUT_VARIABLES.get(name)
            # `ixc`/`icc` carry `module=None`: they are the problem statement, not a
            # field, so they are no more resolvable than an undeclared name.
            area = None if declaration is None else declaration.module
        if area is not None:
            return VarPath((GetAttrKey(area), GetAttrKey(name)))
        raise ValueError(
            f"{name!r} resolves to nothing, and the two halves of that are different "
            f"failures. (1) No node in this graph owns or reads it -- it is not part of "
            f"*this* machine's dataflow, and the same name may well be produced on "
            f"another device configuration, so check the machine before the spelling. "
            f"(2) It is not a declared PROCESS input either "
            f"(`vocabulary/input_variables.py`, PROCESS's own `INPUT_VARIABLES`), so it "
            f"cannot be a constraint bound read off the boundary -- which leaves a "
            f"typo, or a quantity an unported model computes, in which case it "
            f"belongs in "
            f"`NON_INPUT_FIELDS` with its `DataStructure` area."
        )


def _bind(fn, resolve, switch_values):
    """`(switch pairs, read names, inputs)` for one ported constraint/objective
    function.
    """
    parameters = list(inspect.signature(fn).parameters)
    static = tuple((p, switch_values[p]) for p in parameters if p in switch_values)
    read = [p for p in parameters if p not in switch_values]
    return static, read, tuple(In(resolve(p)) for p in read)


def constraint_nodes(graph, icc, n_equality, switch_values=None, omit=()):
    """One `ImplementedFunction` per active constraint, plus the equality/inequality
    split.
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
            static, read, inputs = _bind(fn, resolve, switch_values)
        except ValueError as e:
            raise ValueError(
                f"constraint {cid} cannot be assembled: {e}. An `Optimise` missing one "
                f"of PROCESS's active constraints solves a different problem -- pass "
                f"`omit={{{cid}}}` to leave it out on purpose and have it reported"
            ) from e
        condition = prefix_path(
            VarPath((GetAttrKey("constraints"), GetAttrKey(f"c{cid}"))), COND
        )
        nodes[NodePath((GetAttrKey(f"Constraint{cid}"),))] = ImplementedFunction(
            inputs=inputs,
            outputs=(Out(condition),),
            # index 1 of `(residual, normalised_residual, value, bound)` -- see the
            # module docstring.
            fn=_NormalisedResidual(fn, tuple(read), static),
        )
        (equalities if position < n_equality else inequalities).append(condition)
    return nodes, tuple(equalities), tuple(inequalities), omitted


class _NormalisedResidual(eqx.Module):
    """`fn(*args, **switches) -> normalised_residual`, as a module and not a closure."""

    fn: object
    names: tuple
    switches: tuple = ()
    """`((parameter, value), ...)`, the switch arguments frozen at assembly."""

    def __call__(self, *args):
        arguments = dict(zip(self.names, args, strict=True))
        arguments.update(self.switches)
        return self.fn(**arguments)[1]


class _Metric(eqx.Module):
    """`objective_metric(*args, **switches)`, unsigned."""

    fn: object
    names: tuple = ()
    switches: tuple = ()

    def __call__(self, *args):
        if not self.switches:
            return self.fn(*args)
        arguments = dict(zip(self.names, args, strict=True))
        arguments.update(self.switches)
        return self.fn(**arguments)


class _Negate(eqx.Module):
    """`-x`."""

    def __call__(self, metric):
        return -metric


class ObjectiveSelection(eqx.Module):
    """Which figure of merit this run states, and in which direction -- resolved once,
    at the input-parsing boundary, and never re-derived at assembly.
    """

    metric: object
    """The ported `objective_metric_<id>`, already selected."""

    maximise: bool = eqx.field(static=True)
    """`i_figure_merit < 0` in PROCESS's spelling (`objectives.py:54,105` applies
    `np.sign` outside the branch).
    """


def objective_nodes(graph, selection, switch_values=None):
    """The node(s) computing this run's figure of merit, and the `VarPath` `Optimise`
    minimises.
    """
    switch_values = REFERENCE_SWITCH_VALUES if switch_values is None else switch_values
    resolve = _Resolver(graph)
    static, read, inputs = _bind(selection.metric, resolve, switch_values)
    objective = prefix_path(VarPath((GetAttrKey("numerics"), GetAttrKey("objf"))), COND)
    metric = (
        prefix_path(VarPath((GetAttrKey("numerics"), GetAttrKey("objf"))), METRIC)
        if selection.maximise
        else objective
    )
    nodes = {
        NodePath((GetAttrKey("Objective"),)): ImplementedFunction(
            inputs=inputs,
            outputs=(Out(metric),),
            fn=_Metric(selection.metric, tuple(read), static),
        )
    }
    if selection.maximise:
        nodes[NodePath((GetAttrKey("ObjectiveNegated"),))] = ImplementedFunction(
            inputs=(In(metric),),
            outputs=(Out(objective),),
            fn=_Negate(),
        )
    return nodes, objective


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
    """`graph` with the constraint nodes, the objective node and one `Optimise`
    inserted.
    """
    design = tuple(iteration_variable_path(i) for i in ixc)
    nodes, equalities, inequalities, omitted = constraint_nodes(
        graph, icc, n_equality, switch_values, omit
    )
    from functional_process.cottax.indat import objective_selection  # noqa: PLC0415

    objective_built, objective = objective_nodes(
        graph, objective_selection(i_figure_merit), switch_values
    )
    nodes.update(objective_built)
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


@dataclasses.dataclass(frozen=True)
class FixedPointResidual:
    """One `FixedPoint`'s residual Jacobian `d(g(u) - u)/du` at a point -- or the reason
    it could not be formed.
    """

    problem: NodePath
    jacobian: np.ndarray | None
    """`(n, n)` over the block's unknowns, flattened and concatenated, with the identity
    subtracted; `None` exactly when `undetectable` is set.
    """
    undetectable: str | None
    """The exception that stopped the measurement, `f"{type}: {message}"`, or `None`."""

    @property
    def rank(self) -> int | None:
        """Numerical rank of `jacobian`, or `None` if it could not be formed."""
        if self.jacobian is None:
            return None
        return int(np.linalg.matrix_rank(self.jacobian))

    @property
    def columns(self) -> int | None:
        """How many flattened unknowns the block owns -- the rank a well-posed residual
        block has.
        """
        return None if self.jacobian is None else int(self.jacobian.shape[1])

    @property
    def degenerate(self) -> bool:
        """`True` only when the residual is *identically* zero here -- the strongest
        form of rank deficiency, and the only one `sand_schedule` can act on by dropping
        the problem.
        """
        return self.jacobian is not None and bool(np.allclose(self.jacobian, 0.0))


def fixed_point_residuals(graph, env, problems=None):
    """`d(g(u) - u)/du` for every `FixedPoint` in `graph`, differentiated at `env`."""
    from cottax.evaluate import _run_acyclic

    if problems is None:
        problems = tuple(n for n in graph.declared if is_fixed_point(graph[n]))
    residuals = []
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
        inside = graph.ancestors(set(producers.values()))
        body = graph.subgraph([n for n in inside if n not in graph.declared])

        def residual(flat, _body=body, _owns=owns, _reads=reads, _unravel=None):
            values = dict(env)
            values.update(zip(_owns, _unravel(flat), strict=True))
            out = _run_acyclic(_body, values)
            return jnp.concatenate([jnp.ravel(jnp.asarray(out[r])) for r in _reads])

        try:
            start, unravel = ravel_pytree([jnp.asarray(env[v]) for v in owns])
            # `np.array`, not `np.asarray`: a JAX array converts to a **read-only** view,
            # and the identity subtraction below is in place.
            jacobian = np.array(
                # Jitted: eagerly this is six `jacfwd`s over `_run_acyclic` bodies, one
                # XLA compile per `jnp` primitive -- 45.9 s / 1035 compiles against
                # 4.0 s / 6 (`_audit/next_steps.md` §24.11). `env` is a closure, not an
                # argument, so no `VarPath` is flattened and no antichain question arises.
                eqx.filter_jit(
                    jax.jacfwd(functools.partial(residual, _unravel=unravel))
                )(start),
                dtype=float,
            ).reshape(-1, start.size)
        except Exception as error:  # noqa: BLE001 -- recorded, not swallowed
            reason = f"{type(error).__name__}: {error}"
            residuals.append(FixedPointResidual(problem, None, reason))
            continue
        if jacobian.shape[0] != jacobian.shape[1]:
            # A `FixedPoint`'s conditions are `g(u)`, one per unknown and of the same
            # shape, so this is square by construction -- and if it ever is not, the
            # identity below would be nonsense and the residual is not the thing this
            # function claims to measure. Recorded as the block's own reason rather than
            # raised, because one malformed block must not stop the other measurements.
            residuals.append(
                FixedPointResidual(
                    problem,
                    None,
                    f"ValueError: residual is {jacobian.shape}, not square -- "
                    f"`conditions_of` and `owns` do not correspond element for element",
                )
            )
            continue
        jacobian -= np.eye(start.size)
        residuals.append(FixedPointResidual(problem, jacobian, None))
    return tuple(residuals)


def degenerate_fixed_points(graph, env, problems=None):
    """`FixedPoint` problems whose residual `g(u) - u` is *structurally* zero here."""
    measured = fixed_point_residuals(graph, env, problems)
    undetectable = [r for r in measured if r.undetectable is not None]
    if undetectable:
        detail = "; ".join(
            f"{r.problem.path_str()} ({r.undetectable})" for r in undetectable
        )
        raise ValueError(
            f"cannot tell whether {len(undetectable)} of {len(measured)} fixed "
            f"point(s) are degenerate, so cannot report the rest healthy either: "
            f"{detail}"
        )
    return tuple(r.problem for r in measured if r.degenerate)


def array_valued_problems(graph, env, problems=None):
    """Declared problems owning a **non-scalar** unknown at `env`'s own values -- the
    ones today's SAND layer cannot absorb, detected rather than listed.
    """
    if problems is None:
        problems = tuple(n for n in graph.declared if is_fixed_point(graph[n]))
    return tuple(
        problem
        for problem in problems
        if any(
            unknown in env and jnp.ndim(jnp.asarray(env[unknown])) > 0
            for unknown in graph[problem].owns
        )
    )


def sand_graph(graph, skip=(), keep=()):
    """`graph` with every `FixedPoint` (bar `skip`/`keep`) residualised and every
    problem (bar `keep`) combined into one `^problem.sand`.
    """
    keep = frozenset(keep)
    plan = Plan(graph)
    residualised = []
    for problem in graph.declared:
        if problem in skip or problem in keep:
            continue
        if not is_fixed_point(graph[problem]):
            continue
        plan = plan + Residualise(problem)
        residualised.append(problem)
    # The optimiser first: `+` concatenates and is order-preserving now (it used to
    # absorb from whichever side it was written on), so the design variables lead the
    # combined unknowns -- which `sand_harness`'s Schur reduction indexes positionally.
    folding = [p for p in plan.graph.declared if p not in keep]
    folding.sort(key=lambda p: not is_optimise(plan.graph[p]))
    plan = plan + Combine(NodePath((GetAttrKey("sand"),)), tuple(folding))
    return plan.graph, tuple(residualised)


def constraints_outside_block(graph):
    """Active constraints whose node falls **outside** the combined problem's own SCC
    block -- `{constraint id: NodePath}` -- which today's evaluation seam cannot carry.
    """
    blocking = Blocking.scc(graph)
    # The `Optimise`'s own block, found by walking `blocks` for the node rather than by
    # asking `blocking.problems`. That property raises on a block declaring two
    # problems, which is exactly the shape `sand_graph(keep=...)` leaves behind, and
    # this question -- *which constraints are outside the optimiser's block* -- has the
    # same answer either way. Nothing else about the check changes.
    optimise = next(p for p, d in graph.definitions.items() if is_optimise(d))
    problem_block = next(
        frozenset(nodes) for nodes in blocking.blocks if optimise in nodes
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
    nest=False,
    inner_drivers=None,
    optimiser=None,
):
    """A `Schedule` for `graph`'s single `^problem.sand`, answered by `driver`."""
    optimise = next(p for p, d in graph.definitions.items() if is_optimise(d))
    drivers = default_drivers(
        graph,
        bounds=bounds,
        callback=callback,
        condition_scale=condition_scale,
        max_iter=max_iter,
        **({} if optimiser is None else {"optimiser": optimiser}),
    )
    if driver is not None:
        drivers[optimise] = driver
    drivers.update(inner_drivers or {})
    # Drivers go into the graph (`Assign`), and `schedule_for` reads them from there.
    assigned = assign_drivers(graph, drivers)
    # Nesting is an op on the *graph* now, not a call on the blocking: which statement's
    # iteration answers which is recorded in `Graph.within`, and `Blocking` reads it.
    if nest:
        assigned = (assigned + NestInside(optimise)).graph
    return Schedule(Blocking.scc(assigned))


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
    """
    from cottax.plan import Delete

    driven = cut_graph(graph)
    degenerate = degenerate_fixed_points(driven, env)
    if degenerate:
        driven = Delete(degenerate).apply(driven)
    # By keyword, not by position: `optimise_graph` takes `driver` between
    # `i_figure_merit` and `switch_values`, so the positional call this used to make
    # handed the switch dict to `Assign` as a driver and `omit` as the switch values.
    # The failure was invisible until the first caller outside the harness (the
    # spherical-tokamak assembly probe, 2026-08-30) -- `sand_harness.mda_env`'s own
    # call has always used keywords, which is why nothing caught it.
    with_problem, problem_name, report = optimise_graph(
        driven,
        ixc,
        icc,
        n_equality,
        i_figure_merit,
        switch_values=switch_values,
        omit=omit,
    )
    combined, residualised = sand_graph(with_problem)
    report["degenerate"] = degenerate
    report["residualised"] = residualised
    return combined, problem_name, report
