import os
HERE = os.path.dirname(os.path.abspath(__file__))
import os
exec(open(HERE + "/two_drivers_solve.py").read())
from functional_process.cottax.run_sand_harness import _why_no_step
plasma = next(d for d in drives if "plasma" in d.problem.spelling)
context = {v: out[v] for v in plasma.context}          # what the magnet driver + calls handed it
start = {u: out[u] if u not in design else seeded[u] for u in plasma.unknowns}
for u in plasma.unknowns: start[u] = seeded.get(u, out.get(u))
stuck = _why_no_step(plasma, context, start)
print("violated-and-constant conditions at the plasma start:", stuck)
cm = plasma.condition_map(context)
vals = np.asarray([float(np.asarray(v)) for v in cm(*[jnp.asarray(start[u]) for u in plasma.unknowns])])
rows = np.asarray(jax.jacfwd(lambda *x: jnp.stack([jnp.asarray(v) for v in cm(*x)]))(*[jnp.asarray(start[u]) for u in plasma.unknowns])).reshape(len(vals), -1)
for c, v, r in zip(plasma.conditions, vals, rows):
    print(f"   {c.spelling:48s} value={v:+.3e}  |grad|={np.linalg.norm(r):.2e}")
print("handed from magnet: rmajor=%.3f bt=%.3f" % (float(out[iteration_variable_path(3)]), float(out[iteration_variable_path(2)])), "bounds:", [b for b in ref.bounds if b[0] in {iteration_variable_path(2), iteration_variable_path(3)}])
