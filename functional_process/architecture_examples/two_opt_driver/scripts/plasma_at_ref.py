import os
HERE = os.path.dirname(os.path.abspath(__file__))
import os
exec(open(HERE + "/two_drivers_solve.py").read().split("t0 = time.perf_counter()")[0])
from functional_process.cottax.run_cold_matrix import build_sand, solve_sand
from functional_process.cottax.mda import guess_sources as gs
b = build_sand(ref, machine_graph, sw)
r = solve_sand(b, ref, machine_graph, ref.cold)
xref = dict(zip(ref.ixc, r["_x"]))
print("reference SAND: status", r.get("status"), "iterations", r.get("iterations"), "objf", r.get("objf"), "x =", {k: round(v, 4) for k, v in xref.items()})

# plasma-only graph: magnet variables are boundary inputs, fixed at the reference SAND answer
plasma_only = fold(build_graph({"OptPlasma": ([4, 6, 10, 109], coe, [2, 16, 8, 17, 18, 24, 62, 67])}))
tr = []
drv = default_drivers(plasma_only, bounds=ref.bounds, max_iter=200)
sch0 = Schedule(Blocking.scc(assign_drivers(plasma_only, drv)))
pd = next(s for s in sch0.steps if isinstance(s, Drive) and "plasma" in s.problem.spelling)
drv[pd.problem] = dataclasses.replace(drv[pd.problem], callback=_recorder(tr), condition_scale=residual_condition_scales(pd, env))
sch = Schedule(Blocking.scc(assign_drivers(plasma_only, drv)))
pd = next(s for s in sch.steps if isinstance(s, Drive) and "plasma" in s.problem.spelling)
# MDA at the reference machine, to seed the lifted unknowns consistently
_, runnable, msched, mrun = __import__("functional_process.cottax.sand_harness", fromlist=["mda_schedule"]).mda_schedule(machine_graph)
mg = gs(runnable)
menv = {}
for v in msched.inputs:
    src = mg.get(v, v)
    if src in {iteration_variable_path(i) for i in ref.ixc}: menv[v] = jnp.asarray(xref[next(i for i in ref.ixc if iteration_variable_path(i) == src)])
    elif src in env: menv[v] = env[src]
    else:
        try: menv[v] = jnp.asarray(ground_truth(ref.cold, src))
        except Exception: menv[v] = jnp.asarray(0.0)
from cottax.names import PathMap
mda_ref = dict(mrun(PathMap(menv)))
s_env, _ = _seed(sch, pd, ref.cold, mda_ref, design={iteration_variable_path(i) for i in (4, 6, 10, 109)})
for i in (2, 3, 56, 59): s_env[iteration_variable_path(i)] = jnp.asarray(xref[i])
g2 = gs(sch.blocking.graph)
for v in sch.inputs:
    src = g2.get(v)
    if src is not None:
        key = src if src in mda_ref else next((k for k in mda_ref if k.spelling == src.spelling.replace("^hat", "")), None)
        if key is not None: s_env[v] = given_start(src, mda_ref[key])
out2 = run_schedule(sch, _inputs_only(sch, s_env), whole=False)
it, objf, max_eq, min_ie = _trace_tail(tr)
print(f"plasma-only at the reference machine: status={int(np.asarray(mdf.verdict(out2, Status, pd.problem)))} iterations={it} objf={objf} max|eq|={max_eq} min_ie={min_ie}")
print("   x =", {i: round(float(np.asarray(out2[iteration_variable_path(i)])), 4) for i in ref.ixc}, " reference x =", {k: round(v, 4) for k, v in xref.items()})
