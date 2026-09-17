import os
HERE = os.path.dirname(os.path.abspath(__file__))
import os, time
os.environ["BT_MIN"] = "5.0"
src = open(HERE + "/two_drivers_btmin.py").read()
exec(src.replace("exec(tail.replace(", "exec(tail.split('t0 = time.perf_counter()')[0].replace("))
from functional_process.cottax.run_cold_matrix import build_sand, solve_sand
from functional_process.cottax.mda import guess_sources as gs
from cottax.names import unminted
# stage 1+2: the two sequential drivers
t0 = time.perf_counter(); out = run_schedule(solve, _inputs_only(solve, seeded), whole=False); t_seq = time.perf_counter() - t0
print("two-driver: %.2fs, objf %.4f, iterations %s" % (t_seq, float(out[coe]), {d.problem.spelling.split('.')[-1]: len(traces[d.problem]) for d in drives}))
# stage 3: full SAND, cold (as the harness does it) vs warm-started from `out`
bs = build_sand(ref, machine_graph, sw)
r_cold = solve_sand(bs, ref, machine_graph, ref.cold)
print("SAND cold:  %s, %d iterations, objf %.6f, %.2fs" % (r_cold["status"], r_cold["iterations"], r_cold["objf"], r_cold["_seconds_total"]))
sched, drive = bs.solve_schedule, bs.drive
stage_env = mda_env(ref, graph=machine_graph, data=ref.cold)[1]
base, _ = _seed(sched, drive, ref.cold, stage_env, design=bs.design_paths)
guess = gs(sched.blocking.graph)
warm = dict(base); n = 0
for v in sched.inputs:
    s = guess.get(v, v)
    key = s if s in out else next((k for k in out if k.spelling == s.spelling), None)
    if key is None and s.spelling.startswith("^hat."):
        key = next((k for k in out if k.spelling == s.spelling), None)
    if key is not None: warm[v] = out[key]; n += 1
print(f"warm start: {n} of {len(sched.inputs)} schedule inputs taken from the two-driver answer")
bs.trace.clear()
t0 = time.perf_counter(); o2 = run_schedule(sched, _inputs_only(sched, warm), whole=False); t_w = time.perf_counter() - t0
it, objf, max_eq, min_ie = _trace_tail(bs.trace)
print("SAND warm:  status %d, %d iterations, objf %.6f, %.2fs (+ %.2fs for the two-driver stage)" % (int(np.asarray(mdf.verdict(o2, Status, drive.problem))), it, objf, t_w, t_seq))
print("   x =", {i: round(float(np.asarray(o2[iteration_variable_path(i)])), 4) for i in ref.ixc})

# fair timing: every program compiled, three repeats each
def t_seq():
    for t in traces.values(): t.clear()
    t0 = time.perf_counter(); run_schedule(solve, _inputs_only(solve, seeded), whole=False); return time.perf_counter() - t0
def t_cold():
    bs.trace.clear(); t0 = time.perf_counter(); run_schedule(sched, _inputs_only(sched, base), whole=False); return time.perf_counter() - t0, len(bs.trace)
def t_warm():
    bs.trace.clear(); t0 = time.perf_counter(); run_schedule(sched, _inputs_only(sched, warm), whole=False); return time.perf_counter() - t0, len(bs.trace)
t_cold(); t_warm()
print("compiled, 3 repeats:")
print("   two-driver stage      :", [round(t_seq(), 3) for _ in range(3)], "s")
print("   SAND from cold        :", [(round(t, 3), n) for t, n in (t_cold() for _ in range(3))], "(s, iterations)")
print("   SAND from two-driver  :", [(round(t, 3), n) for t, n in (t_warm() for _ in range(3))], "(s, iterations)")
