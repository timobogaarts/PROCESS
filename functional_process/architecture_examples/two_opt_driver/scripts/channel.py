import os
HERE = os.path.dirname(os.path.abspath(__file__))
import os
exec(open(HERE + "/plasma_at_ref.py").read().split("out2 = run_schedule")[0])
def mda_at(machine):
    m = dict(menv)
    for v in msched.inputs:
        s = mg.get(v, v)
        for i, val in machine.items():
            if s == iteration_variable_path(i): m[v] = jnp.asarray(val)
    return dict(mrun(PathMap(m)))
A = mda_at({2: 4.7164, 3: 26.6445, 56: 1.994571402655538, 59: 0.3006485859172123})
B = mda_at({2: 4.7164, 3: 26.6445, 56: 31.8104, 59: 0.7177})
ctx = set(pd.context)
diffs = []
for k in A:
    if k in B and k in ctx:
        a, b = np.asarray(A[k], float), np.asarray(B[k], float)
        if a.shape == b.shape and a.size == 1 and not np.isclose(a, b, rtol=1e-6):
            diffs.append((k.spelling, float(a), float(b)))
print(f"{len(diffs)} plasma-context inputs differ between (tdmptf 2.0, fcutfsu 0.30) and (31.8, 0.72):")
for s, a, b in sorted(diffs): print(f"   {s:55s} {a:14.6g} -> {b:14.6g}")
for name in (".heat_transport.p_plant_electric_net_mw", ".heat_transport.p_cryo_plant_electric_mw", ".heat_transport.p_plant_electric_gross_mw", ".tfcoil.p_cryo_tf_mw", ".physics.beta_total_vol_avg", ".physics.p_fusion_total_mw"):
    k = next((k for k in A if k.spelling == name), None)
    if k: print(f"   [{name}] {float(A[k]):.6g} -> {float(B[k]):.6g}")
