import os
HERE = os.path.dirname(os.path.abspath(__file__))
import os, sys, time, dataclasses
sys.argv = [sys.argv[0]]
exec(open(HERE + "/two_drivers2.py").read().split("F = xDSMFormatterFlat()")[0])
import numpy as np, jax.numpy as jnp
from functional_process.cottax.sand_harness import mda_env, run_schedule, ground_truth
from functional_process.cottax.run_sand_harness import _seed, _inputs_only
from functional_process.cottax.run_cold_matrix import _recorder, _trace_tail
from functional_process.cottax.sand import residual_condition_scales
from functional_process.cottax import mdf
from functional_process.cottax.core.solver.drivers import Status, VMCON_NON_FINITE

traces = {}
drivers = default_drivers(folded, bounds=ref.bounds, max_iter=200)
probe = Schedule(Blocking.scc(assign_drivers(folded, drivers)))
for step in probe.steps:
    if isinstance(step, Drive) and is_optimise(getattr(folded[step.problem], "problem", folded[step.problem])):
        traces[step.problem] = []
        drivers[step.problem] = dataclasses.replace(
            drivers[step.problem],
            callback=_recorder(traces[step.problem]),
            condition_scale=residual_condition_scales(step, env),
        )
solve = Schedule(Blocking.scc(assign_drivers(folded, drivers)))
drives = [s for s in solve.steps if isinstance(s, Drive) and s.problem in traces]
design = {iteration_variable_path(i) for i in ref.ixc}
seeded = {}
for d in drives:
    e, borrowed = _seed(solve, d, ref.cold, env, design=design)
    seeded.update(e)
from functional_process.cottax.mda import guess_sources, given_start
guesses = guess_sources(solve.blocking.graph)
reseeded = []
for v in solve.inputs:
    src = guesses.get(v)
    if src is not None and src not in design and src in env:
        seeded[v] = given_start(src, env[src]); reseeded.append(v.spelling)
print("guess ports re-seeded from the MDA env:", reseeded)
t0 = time.perf_counter()
try:
    out = run_schedule(solve, _inputs_only(solve, seeded), whole=False)
except Exception as e:
    print("FAILED:", type(e).__name__, str(e)[:120].replace("\n"," "))
    for p, t in traces.items(): print(f"   {p.spelling}: {len(t)} VMCON iterations recorded; last={t[-1] if t else None}")
    raise SystemExit(1)
print(f"\nran in {time.perf_counter()-t0:.1f}s")
for d in drives:
    st = mdf.verdict(out, Status, d.problem)
    it, objf, max_eq, min_ie = _trace_tail(traces[d.problem])
    print(f"{d.problem.spelling:25s} status={None if st is None else int(np.asarray(st))} iterations={it} objf={objf} max|eq|={max_eq} min_ie={min_ie}")
print("design:", {i: float(np.asarray(out[iteration_variable_path(i)])) for i in ref.ixc})
print("COE objective (^cond.numerics.objf):", float(np.asarray(out[coe])))
print("rmajor objective:", float(np.asarray(out[rmajor_obj])))
print("constraints:", {c: round(float(np.asarray(out[cond(c)])), 4) for c in ref.icc})
