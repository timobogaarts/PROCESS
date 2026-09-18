import os
HERE = os.path.dirname(os.path.abspath(__file__))
import os, time
os.environ["BT_MIN"] = "5.0"
src = open(HERE + "/two_drivers_btmin.py").read()
exec(src.replace("exec(tail.replace(", "exec(tail.split('t0 = time.perf_counter()')[0].replace("))   # build + seed only, no solve
import gc
from functional_process.cottax.run_cold_matrix import build_sand, solve_sand

def timed_two_driver():
    for t in traces.values(): t.clear()
    t0 = time.perf_counter(); o = run_schedule(solve, _inputs_only(solve, seeded), whole=False); dt = time.perf_counter() - t0
    return dt, {d.problem.spelling.split(".")[-1]: len(traces[d.problem]) for d in drives}, float(np.asarray(o[coe]))

# two-driver: cold (first call compiles), then warm x3
o_run = run_schedule(solve, _inputs_only(solve, seeded), whole=False)
cold_two = timed_two_driver()
warm_two = [timed_two_driver() for _ in range(3)]
print('two-driver cold/warm done')
# SAND: build (compiles on first solve), cold then warm x3
t0 = time.perf_counter(); bs = build_sand(ref, machine_graph, sw); t_build = time.perf_counter() - t0
def timed_sand():
    t0 = time.perf_counter(); r = solve_sand(bs, ref, machine_graph, ref.cold); dt = time.perf_counter() - t0
    return dt, r["iterations"], r["objf"], r.get("_seconds_total")
cold_sand = timed_sand()
warm_sand = [timed_sand() for _ in range(3)]
print("two-driver  cold: %.2fs  iterations=%s  objf=%.4f" % (cold_two[0], cold_two[1], cold_two[2]))
print("two-driver  warm: %s  iterations=%s" % ([round(w[0], 3) for w in warm_two], warm_two[-1][1]))
print("SAND build (assembly, no solve): %.2fs" % t_build)
print("SAND        cold: %.2fs  iterations=%s  objf=%.4f" % (cold_sand[0], cold_sand[1], cold_sand[2]))
print("SAND        warm: %s  iterations=%s" % ([round(w[0], 3) for w in warm_sand], warm_sand[-1][1]))
# per-iteration block cost: one condition-map evaluation + jacobian per drive, warm
import jax
for label, d in [("magnet", drives[0]), ("plasma", drives[1])]:
    ctx = {v: o_run[v] for v in d.context}
    cm = d.condition_map(ctx); xs = [jnp.asarray(o_run[u]) for u in d.unknowns]
    f = jax.jit(lambda *x: jnp.stack([jnp.asarray(v) for v in cm(*x)])); J = jax.jit(jax.jacfwd(lambda *x: jnp.stack([jnp.asarray(v) for v in cm(*x)])))
    f(*xs); J(*xs)
    t0 = time.perf_counter(); [jax.block_until_ready(f(*xs)) for _ in range(20)]; tf = (time.perf_counter()-t0)/20
    t0 = time.perf_counter(); [jax.block_until_ready(J(*xs)) for _ in range(20)]; tj = (time.perf_counter()-t0)/20
    print(f"   {label}: {len(d.nodes)} nodes, {len(d.unknowns)} unknowns x {len(d.conditions)} conditions -- values {tf*1e3:.2f} ms, jacfwd {tj*1e3:.2f} ms (warm, jitted)")
sd = bs.drive
ctx = {v: (env[v] if v in env else seeded.get(v, jnp.asarray(0.0))) for v in sd.context}
cm = sd.condition_map(ctx); xs = [jnp.asarray(env[u] if u in env else seeded.get(u, o_run.get(u, 0.0))) for u in sd.unknowns]
f = jax.jit(lambda *x: jnp.stack([jnp.asarray(v) for v in cm(*x)])); J = jax.jit(jax.jacfwd(lambda *x: jnp.stack([jnp.asarray(v) for v in cm(*x)])))
f(*xs); J(*xs)
t0 = time.perf_counter(); [jax.block_until_ready(f(*xs)) for _ in range(20)]; tf = (time.perf_counter()-t0)/20
t0 = time.perf_counter(); [jax.block_until_ready(J(*xs)) for _ in range(20)]; tj = (time.perf_counter()-t0)/20
print(f"   SAND: {len(sd.nodes)} nodes, {len(sd.unknowns)} unknowns x {len(sd.conditions)} conditions -- values {tf*1e3:.2f} ms, jacfwd {tj*1e3:.2f} ms (warm, jitted)")
