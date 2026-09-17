import os
HERE = os.path.dirname(os.path.abspath(__file__))
import os
exec(open(HERE + "/two_drivers_solve.py").read().split("t0 = time.perf_counter()")[0])
from cottax.names import unminted
gp = [v for v in solve.inputs if "guess" in v.spelling]
print("guess ports:", [(v.spelling, guesses.get(v).spelling if guesses.get(v) else None, float(np.asarray(seeded[v]))) for v in gp])
dd = [v for v in env if v.spelling == ".vacuum.d_duct"]
print("d_duct in MDA env:", [(v.spelling, float(np.asarray(env[v]))) for v in dd])
print("d_duct in cold:", ground_truth(ref.cold, dd[0]) if dd else None)
# how did mda_env seed it?
_, runnable, msched, _ = __import__("functional_process.cottax.sand_harness", fromlist=["mda_schedule"]).mda_schedule(machine_graph)
mg = guess_sources(runnable)
print("mda schedule guess for d_duct:", [(v.spelling, mg[v].spelling) for v in msched.inputs if "d_duct" in v.spelling])
from functional_process.cottax.mda import GIVEN_STARTS
print("GIVEN_STARTS:", {k.spelling if hasattr(k,'spelling') else k: v for k, v in GIVEN_STARTS.items()})
