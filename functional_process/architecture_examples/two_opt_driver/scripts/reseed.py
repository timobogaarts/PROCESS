import os
HERE = os.path.dirname(os.path.abspath(__file__))
import os
exec(open(HERE + "/two_drivers_solve.py").read())
from functional_process.cottax.sand_harness import mda_schedule, _schedule_runners
from functional_process.cottax.run_sand_harness import _why_no_step
plasma = next(d for d in drives if "plasma" in d.problem.spelling)
magnet = next(d for d in drives if "magnet" in d.problem.spelling)
G = solve.blocking.graph
# (a) the residual node: what it reads, and who owns those reads
res = [n for n in plasma.nodes if "f_ster_div_single" in n.spelling]
for n in res:
    node = G[n]
    print(f"{n.spelling}: reads {[r.spelling for r in node.reads]} owns {[o.spelling for o in node.owns]}")
    for r in node.reads:
        o = G.owners.get(r); print(f"     {r.spelling:45s} owned by {o.spelling if o else 'BOUNDARY'}  in plasma block: {o in set(plasma.nodes) if o else '-'}  in magnet block: {o in set(magnet.nodes) if o else '-'}")
# (b) re-seed the plasma block's lifted unknowns from an MDA at the magnet's answer, then run the plasma drive
_, runnable, msched, mrun = mda_schedule(machine_graph)
from functional_process.cottax.mda import guess_sources as gs
mg = gs(runnable)
menv = {}
for v in msched.inputs:
    src = mg.get(v, v)
    if src in out: menv[v] = out[src]                     # the magnet's design, and everything it produced
    elif v in seeded: menv[v] = seeded[v]
    else:
        try: menv[v] = jnp.asarray(ground_truth(ref.cold, src))
        except Exception: menv[v] = jnp.asarray(0.0)
from cottax.names import PathMap
mda_at_magnet = dict(mrun(PathMap(menv)))
print("MDA at magnet answer: rmajor=%.3f, f_ster_div_single=%s" % (float(mda_at_magnet[iteration_variable_path(3)]), float(np.asarray(mda_at_magnet[next(v for v in mda_at_magnet if v.spelling=='.fwbs.f_ster_div_single')]))))
# run schedule up to and including the magnet, then reseed the plasma guesses and continue
runners = _schedule_runners(solve, fuse_upstream=False)
groups, g_ = [], []
for s in solve.steps:
    if isinstance(s, Drive):
        if g_: groups.append(g_); g_ = []
        groups.append([s])
    else: g_.append(s)
if g_: groups.append(g_)
env2 = dict(_inputs_only(solve, seeded))
for runner, grp in zip(runners, groups):
    if grp[0] is plasma:
        for v in plasma.subgraph.boundary_inputs:
            src = guesses.get(v)
            if src is not None and src not in design:
                key = src if src in mda_at_magnet else next((k for k in mda_at_magnet if k.spelling == src.spelling.replace("^hat","")), None)
                if key is not None: env2[v] = mda_at_magnet[key]; print("   reseeded", v.spelling, "<-", key.spelling, float(np.asarray(env2[v])))
        traces[plasma.problem].clear()
    env2 = runner(env2)
it, objf, max_eq, min_ie = _trace_tail(traces[plasma.problem])
st = mdf.verdict(env2, Status, plasma.problem)
print(f"plasma after reseed: status={int(np.asarray(st))} iterations={it} objf={objf} max|eq|={max_eq} min_ie={min_ie}")
print("design:", {i: float(np.asarray(env2[iteration_variable_path(i)])) for i in ref.ixc})
print("constraints:", {c: round(float(np.asarray(env2[cond(c)])), 4) for c in ref.icc})
