import os
HERE = os.path.dirname(os.path.abspath(__file__))
import os
exec(open(HERE + "/reseed.py").read().split("# run schedule up to and including")[0])
from functional_process.cottax.run_sand_harness import _why_no_step
ctx = {v: out[v] for v in plasma.context}
hat = next(u for u in plasma.unknowns if "f_ster" in u.spelling)
start = {u: seeded.get(u, out.get(u)) for u in plasma.unknowns}
start[hat] = jnp.asarray(0.01477607429341634)
cm = plasma.condition_map(ctx)
def f(h):
    xs = [jnp.asarray(start[u]) if u is not hat else h for u in plasma.unknowns]
    return jnp.stack([jnp.asarray(v) for v in cm(*xs)])
v0 = np.asarray(f(jnp.asarray(0.01477607429341634))); v1 = np.asarray(f(jnp.asarray(0.02)))
idx = [i for i, c in enumerate(plasma.conditions) if "f_ster" in c.spelling][0]
print("f_ster residual: value at hat=0.014776 -> %.3e ; at hat=0.02 -> %.3e ; jacfwd wrt hat = %s" % (v0[idx], v1[idx], np.asarray(jax.jacfwd(f)(jnp.asarray(0.01477607429341634)))[idx]))
print("all conditions that change with hat:", [(plasma.conditions[i].spelling, float(v1[i]-v0[i])) for i in range(len(v0)) if abs(v1[i]-v0[i]) > 1e-12])
print("stuck after reseed:", _why_no_step(plasma, ctx, start))
