import os
HERE = os.path.dirname(os.path.abspath(__file__))
import os
exec(open(HERE + "/two_drivers_solve.py").read())
from cottax.problem import conditions_of
from functional_process.cottax.sand import reference_problem, sand_schedule
from functional_process.cottax.visualization.grouping import render_grouped_dsm_html, structure_order
from cottax.interfaces.pytree_namespace_module import xDSMFormatterFlat
F = xDSMFormatterFlat()
render_grouped_dsm_html(solve.blocking, order=structure_order(solve.blocking),
    title="stellarator_helias -- two sequential Optimise, the EXECUTING (driven) graph, SCC order",
    file_name="dsm_two_drivers_executing", outdir=OUT, write=True, formatter=F)

DESC = {2:"global power balance", 16:"net electric power >= target", 24:"beta upper limit", 8:"neutron wall load", 17:"radiation power", 18:"divertor heat load", 67:"radiation wall load", 82:"toroidal build (coil-coil gap)", 83:"radial build (blanket space)", 62:"He thermal (alpha confinement)", 32:"TF conductor stress", 34:"TF dump voltage", 35:"TF quench temperature", 65:"VV stress on quench"}
EQ = set(ref.icc[:ref.n_equality])
def describe(drive, graph):
    node = graph[drive.problem]; prob = getattr(node, "problem", node)
    print(f"\n### {drive.problem.spelling}  ({len(drive.nodes)} nodes in block)")
    print("objective:", [o.spelling for o in prob.objectives])
    print("unknowns:")
    for u in drive.unknowns:
        kind = "design (ixc)" if u in design else "lifted unknown (residualised FixedPoint copy)"
        print(f"   {u.spelling:55s} {kind}")
    print("conditions:")
    for c in conditions_of(prob):
        s = c.spelling
        if s.startswith("^cond.constraints.c"):
            cid = int(s.rsplit("c", 1)[1])
            kind = ("consistency (PROCESS equality)" if cid in EQ else "engineering limit (PROCESS inequality)") + f" -- icc {cid}: {DESC.get(cid,'')}"
        else:
            kind = "lifted residual (g(u)-u = 0 of a residualised FixedPoint)"
        print(f"   {s:55s} {kind}")
for d in drives: describe(d, solve.blocking.graph)
combined, name, report = reference_problem(driven, ref.ixc, ref.icc, ref.n_equality, ref.i_figure_merit, env=env, switch_values=sw)
rs = sand_schedule(combined, name, bounds=ref.bounds)
describe(next(s for s in rs.steps if isinstance(s, Drive)), rs.blocking.graph)
