import os
HERE = os.path.dirname(os.path.abspath(__file__))
import sys, os, jax
jax.config.update("jax_enable_x64", True)
from cottax.blocking import Blocking
from cottax.plan import Insert, Plan
from cottax.problem import Optimise, is_optimise, is_fixed_point
from functional_process.cottax.sand import declared
from cottax.spec import NodePath, VarPath
from cottax.nodes import ImplementedFunction
from cottax.names import prefix_path, PathMap
from cottax.rewrites import Residualise, Combine
from cottax.evaluation.schedule import Schedule, Drive, Call
from cottax.interfaces.pytree_namespace_module import xDSMFormatterFlat
from cottax.visualization.ragraph_dsm import render_dsm_html
from cottax.visualization import render_xdsm_html
from jax.tree_util import GetAttrKey
from functional_process.cottax import native
from functional_process.cottax.indat import graph_for, machine_from_indat, switch_values_from_indat, objective_selection
from functional_process.cottax.mda import cut_graph, default_drivers, assign_drivers
from functional_process.cottax.sand import constraint_nodes, objective_nodes, iteration_variable_path, _bind, _Resolver, _Metric, COND, design_bounds
from functional_process.cottax.core.solver import objectives as ported_objectives
from functional_process.vocabulary import FiguresOfMerit
from functional_process.cottax.visualization.grouping import render_grouped_dsm_html, structure_order

OUT = HERE
path = "tests/regression/input_files/stellarator_helias.IN.DAT"
ref = native.native_reference(path)
machine_graph = graph_for(machine_from_indat(path))
from functional_process.cottax.sand_harness import mda_env
from functional_process.cottax.sand import degenerate_fixed_points, array_valued_problems
from cottax.rewrites import Delete
driven, env = mda_env(ref, graph=machine_graph)
_drop = tuple(degenerate_fixed_points(driven, env)) + tuple(array_valued_problems(driven, env))
g = Delete(_drop).apply(driven) if _drop else driven
print("dropped before assembly:", [p.spelling for p in _drop])
sw = switch_values_from_indat(path)
cnodes, eqs, ineqs, _ = constraint_nodes(g, ref.icc, ref.n_equality, sw)
cond = lambda cid: prefix_path(VarPath((GetAttrKey("constraints"), GetAttrKey(f"c{cid}"))), COND)
obj_nodes, coe = objective_nodes(g, objective_selection(ref.i_figure_merit), sw)
metric1 = ported_objectives.OBJECTIVE_METRICS[FiguresOfMerit(1)]
static, read, inputs = _bind(metric1, _Resolver(g), sw)
rmajor_obj = prefix_path(VarPath((GetAttrKey("numerics"), GetAttrKey("objf_rmajor"))), COND)
obj_nodes[NodePath((GetAttrKey("ObjectiveRmajor"),))] = ImplementedFunction(reads=inputs, owns=(rmajor_obj,), fn=_Metric(metric1, tuple(read), static))
EQ = set(ref.icc[:ref.n_equality])

def build_graph(split):
    nodes = dict(cnodes); nodes.update(obj_nodes)
    for name, (ixc, objective, cids) in split.items():
        nodes[NodePath((GetAttrKey(name),))] = Optimise(
            objective=objective,
            unknowns=tuple(iteration_variable_path(i) for i in ixc),
            equalities=tuple(cond(c) for c in cids if c in EQ),
            inequalities=tuple(cond(c) for c in cids if c not in EQ),
        )
    return (Plan(g) + Insert(PathMap(nodes.items()))).graph

def fold(graph):
    plan = Plan(graph)
    for p in declared(graph):
        if is_fixed_point(graph[p]):
            plan = plan + Residualise(p)
    for blk in Blocking.scc(plan.graph).blocks:
        probs = [n for n in blk if n in declared(plan.graph)]
        if len(probs) > 1:
            opt = next(n for n in probs if is_optimise(plan.graph[n]))
            probs.sort(key=lambda n: n != opt)
            plan = plan + Combine(NodePath((GetAttrKey("sand_" + opt.leaf.name.removeprefix("Opt").lower()),)), tuple(probs))
    return plan.graph

SPLIT_A = {"OptMagnet": ([2, 3, 56, 59], rmajor_obj, [32, 34, 35, 65, 82, 83]),
           "OptPlasma": ([4, 6, 10, 109], coe, [2, 16, 8, 17, 18, 24, 62, 67])}
SPLIT_B = {"OptMachine": ([2, 3, 4, 6, 10, 109], coe, [2, 16, 8, 17, 18, 24, 62, 67]),
           "OptTF": ([56, 59], rmajor_obj, [32, 34, 35, 65, 82, 83])}

folded = fold(build_graph(SPLIT_A))
b = Blocking.scc(folded)
print("A) problems ->", [p.spelling for p in b.problems if p is not None])
assigned = assign_drivers(folded, default_drivers(folded, bounds=design_bounds(ref.ixc)))
sched = Schedule(Blocking.scc(assigned))
for i, s in enumerate(sched.steps):
    if isinstance(s, Drive):
        print(f"   step {i:3d}: Drive {s.problem.spelling:40s} nodes={len(s.nodes):3d} unknowns={len(s.unknowns):2d} conditions={len(s.conditions):2d} {type(assigned[s.problem].driver).__name__}")
print(f"   {sum(isinstance(s, Call) for s in sched.steps)} Call steps, {len(sched.steps)} total")

try:
    bb = Blocking.scc(fold(build_graph(SPLIT_B)))
    print("B) problems ->", [p.spelling for p in bb.problems if p is not None])
except Exception as e:
    print(f"B) REFUSED: {type(e).__name__}: {str(e)[:200]}")

F = xDSMFormatterFlat()
blk = Blocking.scc(folded)
render_grouped_dsm_html(blk, order=structure_order(blk), title="stellarator_helias -- two sequential Optimise (magnet, then plasma), SCC order",
                        file_name="dsm_two_drivers", outdir=OUT, write=True, formatter=F)
render_xdsm_html(blk, blocks=True, file_name="xdsm_two_drivers", outdir=OUT, write=True, collapse_names=True, collapse_models=True, formatter=F)
try:
    bu = Blocking.scc(build_graph(SPLIT_B))
    render_grouped_dsm_html(bu, order=structure_order(bu), title="stellarator_helias -- machine vs TF-conductor split: one SCC (refused)",
                            file_name="dsm_split_B_refused", outdir=OUT, write=True, formatter=F)
except Exception as e:
    print(f"DSM B skipped: {type(e).__name__}: {str(e)[:200]}")
print("written:", sorted(f for f in os.listdir(OUT) if f.endswith(".html")))

print("\n-- driven graph --")
ba = Blocking.scc(assigned)
for p, t in zip(ba.problems, ba.problem_types):
    if p is not None:
        n = assigned[p]
        print(f"   {p.spelling:45s} type={t:10s} driven={type(n).__name__:8s} driver={type(getattr(n,'driver',None)).__name__}")
render_grouped_dsm_html(ba, order=structure_order(ba), title="stellarator_helias -- two sequential Optimise, DRIVEN graph, SCC order",
                        file_name="dsm_two_drivers_driven", outdir=OUT, write=True, formatter=F)
