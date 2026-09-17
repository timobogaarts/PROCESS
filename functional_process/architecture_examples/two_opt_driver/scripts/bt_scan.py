import os
HERE = os.path.dirname(os.path.abspath(__file__))
import os
src = open(HERE + "/plasma_at_ref.py").read()
head = src.split("out2 = run_schedule")[0]
exec(head)
xmag = {2: 4.000000002197111, 3: 25.535072711620426, 56: 1.994571402655538, 59: 0.3006485859172123}
def run_at(machine, label):
    menv2 = dict(menv)
    for v in msched.inputs:
        s = mg.get(v, v)
        for i, val in machine.items():
            if s == iteration_variable_path(i): menv2[v] = jnp.asarray(val)
    mda = dict(mrun(PathMap(menv2)))
    e, _ = _seed(sch, pd, ref.cold, mda, design={iteration_variable_path(i) for i in (4, 6, 10, 109)})
    for i, val in machine.items(): e[iteration_variable_path(i)] = jnp.asarray(val)
    for v in sch.inputs:
        s = g2.get(v)
        if s is not None:
            key = s if s in mda else next((k for k in mda if k.spelling == s.spelling.replace("^hat", "")), None)
            if key is not None: e[v] = given_start(s, mda[key])
    tr.clear()
    o = run_schedule(sch, _inputs_only(sch, e), whole=False)
    it, objf, max_eq, min_ie = _trace_tail(tr)
    st = int(np.asarray(mdf.verdict(o, Status, pd.problem)))
    pnet = next((float(np.asarray(o[k])) for k in o if k.spelling == ".heat_transport.p_plant_electric_net_mw"), None)
    pfus = next((float(np.asarray(o[k])) for k in o if k.spelling == ".physics.p_fusion_total_mw"), None)
    print(f"{label:45s} status={st} it={it:3d} objf={objf} c2={float(o[cond(2)]):+.3f} c16={float(o[cond(16)]):+.3f} c24={float(o[cond(24)]):+.3f}  P_fus={pfus} P_net={pnet}")
    print("      x =", {i: round(float(np.asarray(o[iteration_variable_path(i)])), 4) for i in (4, 6, 10, 109)})
run_at(xmag, "magnet answer (rmajor 25.5, bt 4.00)")
run_at({**xmag, 2: 4.7164}, "magnet answer but bt 4.72")
run_at({**xmag, 2: 4.7164, 3: 26.6445}, "magnet answer but bt 4.72, rmajor 26.64")
run_at({**xmag, 2: 5.5}, "magnet answer but bt 5.5")
