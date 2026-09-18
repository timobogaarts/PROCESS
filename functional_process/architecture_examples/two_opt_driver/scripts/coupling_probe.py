import os
HERE = os.path.dirname(os.path.abspath(__file__))
import sys, jax
jax.config.update("jax_enable_x64", True)
from cottax.plan import Insert, Plan
from cottax.names import PathMap
from functional_process.cottax import native
from functional_process.cottax.indat import graph_for, machine_from_indat, switch_values_from_indat, objective_selection
from functional_process.cottax.mda import cut_graph
from functional_process.cottax.sand import constraint_nodes, objective_nodes, iteration_variable_path

path = sys.argv[1]
ref = native.native_reference(path)
g = cut_graph(graph_for(machine_from_indat(path)))
sw = switch_values_from_indat(path)
nodes, eqs, ineqs, omitted = constraint_nodes(g, ref.icc, ref.n_equality, sw)
obj_nodes, objective = objective_nodes(g, objective_selection(ref.i_figure_merit), sw)
nodes.update(obj_nodes)
g2 = (Plan(g) + Insert(PathMap(nodes.items()))).graph
design = {iteration_variable_path(i): i for i in ref.ixc}
owned_design = {v: g2.owners[v] for v in design if v in g2.owners}
print(f"{path}: ixc={ref.ixc} icc={ref.icc} neq={ref.n_equality} objf={ref.i_figure_merit}")
print("design vars OWNED by a node (not boundary):", {design[v]: n.spelling for v, n in owned_design.items()})

def reaches(node):
    anc = set(g2.ancestors([node])) | {node}
    hit = set()
    for n in anc:
        for r in g2[n].reads:
            if r in design: hit.add(design[r])
    return hit

rows = []
for name in g2.nodes:
    leaf = name.leaf.name
    if leaf.startswith("Constraint") or leaf.startswith("Objective"):
        rows.append((leaf, reaches(name)))
w = max(len(r[0]) for r in rows)
hdr = "".join(f"{i:>5}" for i in ref.ixc)
print(f"{'':{w}} {hdr}")
for leaf, hit in sorted(rows, key=lambda r: (not r[0].startswith("Obj"), int(r[0][10:]) if r[0][10:].isdigit() else 0)):
    tag = "eq" if leaf.startswith("Constraint") and int(leaf[10:]) in ref.icc[:ref.n_equality] else ("ob" if leaf.startswith("Obj") else "  ")
    print(f"{leaf:{w}} " + "".join(f"{'  X  ' if i in hit else '  .  '}" for i in ref.ixc), tag)
