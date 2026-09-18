import os
HERE = os.path.dirname(os.path.abspath(__file__))
import os, sys
exec(open(HERE + "/two_drivers_solve.py").read().split("t0 = time.perf_counter()")[0])
from functional_process.cottax.sand_harness import _schedule_runners
from cottax.evaluation.schedule import Call
import numpy as np, jax.numpy as jnp

inputs = _inputs_only(solve, seeded)
zeros = [v.spelling for v, x in inputs.items() if float(np.max(np.abs(np.asarray(x)))) == 0.0]
nonfin = [v.spelling for v, x in inputs.items() if not np.all(np.isfinite(np.asarray(x)))]
print(f"seeded inputs: {len(inputs)}; zero-valued: {len(zeros)} {zeros[:15]}; non-finite: {nonfin}")

# step by step, eager groups so the failing step is named
runners = _schedule_runners(solve, fuse_upstream=False)
steps_iter = iter(solve.steps)
env = dict(inputs)
i = 0
groups = []
g = []
for s in solve.steps:
    if isinstance(s, Drive):
        if g: groups.append(g); g = []
        groups.append([s])
    else: g.append(s)
if g: groups.append(g)
for runner, grp in zip(runners, groups):
    label = grp[0].problem.spelling if isinstance(grp[0], Drive) else f"{len(grp)} calls ({grp[0].nodes[0].spelling} .. {grp[-1].nodes[-1].spelling})"
    try:
        new = runner(env)
    except Exception as e:
        print(f"FAILED at {label}: {type(e).__name__}: {str(e)[:100]}")
        if isinstance(grp[0], Drive):
            d = grp[0]
            print("   drive inputs (context) non-finite:", [v.spelling for v in d.context if v in env and not np.all(np.isfinite(np.asarray(env[v])))])
            print("   drive unknown seeds:", {u.spelling: float(np.asarray(env[u])) if u in env else 'MISSING' for u in d.unknowns})
            print("   guess ports seeded:", {v.spelling: float(np.asarray(env[v])) for v in d.subgraph.boundary_inputs if 'guess' in v.spelling and v in env})
        break
    bad = [v.spelling for v in set(new) - set(env) if not np.all(np.isfinite(np.asarray(new[v])))]
    if bad: print(f"   after {label}: non-finite produced: {bad[:10]}")
    env = new
else:
    print("all steps ran")
