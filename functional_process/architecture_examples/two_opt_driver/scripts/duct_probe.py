import os
HERE = os.path.dirname(os.path.abspath(__file__))
import os
exec(open(HERE + "/two_drivers_solve.py").read().split("t0 = time.perf_counter()")[0])
import numpy as np, jax.numpy as jnp
from functional_process.cottax.sand_harness import _schedule_runners
from functional_process.cottax.sand import reference_problem, sand_schedule
duct = next(s for s in solve.steps if isinstance(s, Drive) and "duct" in s.problem.spelling)
print("duct step index in two-driver schedule:", solve.steps.index(duct))
print("duct context:", {v.spelling: float(np.asarray(seeded[v])) if v in seeded else "MISSING" for v in duct.context})
inputs = _inputs_only(solve, seeded)
runners = _schedule_runners(solve, fuse_upstream=False)
env = dict(inputs)
groups, g = [], []
for s in solve.steps:
    if isinstance(s, Drive):
        if g: groups.append(g); g = []
        groups.append([s])
    else: g.append(s)
if g: groups.append(g)
for runner, grp in zip(runners, groups):
    if grp[0] is duct: break
    env = runner(env)
ctx = {v: env[v] for v in duct.context}
print("duct context at run time:", {v.spelling: float(np.asarray(x)) for v, x in ctx.items()})
cm = duct.condition_map(ctx)
for start in (0.0, 1e-6, 1e-3, 1e-2, 0.1, 1.0):
    try:
        r = cm((jnp.asarray(start),))
        print(f"  residual at d_duct={start}: {np.asarray(r)}")
    except Exception as e:
        print(f"  residual at d_duct={start}: {type(e).__name__}: {str(e)[:80]}")
# where does the reference SAND schedule place the duct root-find?
combined, name, report = reference_problem(driven, ref.ixc, ref.icc, ref.n_equality, ref.i_figure_merit, env=env if False else __import__("functional_process.cottax.sand_harness", fromlist=["mda_env"]).mda_env(ref, graph=machine_graph)[1], switch_values=sw)
rs = sand_schedule(combined, name, bounds=ref.bounds)
for i, s in enumerate(rs.steps):
    if isinstance(s, Drive): print(f"reference SAND step {i}: {s.problem.spelling} nodes={len(s.nodes)}")
