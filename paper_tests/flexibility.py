"""Flexibility analysis: fix the machine, let the operator re-optimise per belief draw.

**The question.** PROCESS's deterministic optimum is one machine. Under uncertainty,
in how many worlds can that machine be *operated* within all its limits, and in how
many can it deliver the power it was built for?

**The formulation** (flexibility analysis, in the Grossmann/Halemane sense: `psi` is the
feasibility function and `P(psi <= 0)` the stochastic flexibility). Per draw, two
phases, and they must stay separate:

1. `psi = min_z max_j g_j(z)`, posed in epigraph form (`min t` s.t. `g_j(z) - t <= 0`).
   `psi <= G_TOL` and `net > 0` is what "operable" means. Solving this first is what
   distinguishes "no operating point exists" from "the optimiser stopped a hair outside
   tolerance" -- conflating them reports 3 % operable instead of 41 %.
2. Where operable, `min coe` subject to `g <= max(G_TOL, psi_row)`, warm-started from
   phase 1. **Not `g <= 0`**: `psi > 0` in 100 % of operable draws (c83 sits at
   +1.618087e-04 and the operator cannot move it), so `{g <= 0}` is empty and SLSQP
   silently returns its iteration-limit point. That bias is ~9 % on coe and ~11 % on
   net power.

**What is fixed and what moves.** Five build numbers are held (B, R, quench time,
copper fraction, and the winding-pack width, the last exposed by
`architectures.lift.lift_winding_pack` so the coil is not silently re-sized per
sample). The operator sets `T_e` and the helium fraction. The density is *not* a knob:
`c2` is closed by it inside the MDA, which for an ignited machine is the ignition
condition -- the power balance is monotone in the density and non-monotonic in the
temperature, so PROCESS's ixc 6 / icc 2 pairing is a numerical necessity, not a
convention. `helias_5b` has zero auxiliary heating (`p_hcd_primary_extra_heat_mw = 0`
in the IN.DAT), so there is nothing else that could close it.

**`--require-net` is the second question.** `c16` is in the file's `icc` but not in
this constraint set. 0 (the default) asks "can it be run at all"; the port's own
nominal, 982.4, asks "can it be run at rated power". Do NOT use the file's literal
1000.0: `c16` is an equality PROCESS's optimum sits exactly on, so it has zero margin,
and the port delivers 982.4 MW there (~1.8 % low, a deliberate divergence) -- requiring
1000 measures the port/PROCESS gap rather than robustness, and reports 0 % operable.

**`--order hfact` matters.** Phase 2 is a local solve of a non-convex problem; walking
draws in Sobol' index order warm-starts each from an unrelated world and strands some at
near-zero net power where `coe ~ 1/net` explodes (mean coe 4.6e19). Ordering by the
dominant belief's quantile fixes it at 0.9x the cost. Operability and `psi` are
unaffected; the coe/net *tails* are not.

**Performance.** ~0.13 s/draw after a ~50 s compile (93 % of the wall time is JAX, not
scipy). Embarrassingly parallel: 96 workers on one Snellius `rome` node do N = 262144 in
~6 min at 95 % efficiency.

## Recipe

    PY=~/miniconda3/envs/process_port/bin/python
    SC=<a scratch dir>

    # cottax's working tree moves; measure against the commit the port tracks
    mkdir -p $SC/cottax_head && git -C ~/jaxgraph archive 6b1d540 src | tar -x -C $SC/cottax_head

    cd ~/PROCESS                       # the model build uses relative IN.DAT paths
    export JAX_PLATFORMS=cpu
    export PYTHONPATH=~/PROCESS:$SC/cottax_head/src:~/PROCESS/paper_tests

    $PY paper_tests/flexibility.py --n 1024 --workers 8 --order hfact       # can it run
    $PY paper_tests/flexibility.py --n 1024 --workers 8 --order hfact \
                                   --require-net 982.4                      # at rated power
    $PY paper_tests/flexibility.py --figures    # from the runs already in out/
    $PY paper_tests/flexibility.py --dsm        # the graph it solves

    # at scale, on Snellius (~6 min each, results land in paper_tests/out/)
    cd ~/PROCESS/paper_tests/cluster
    sbatch --export=ALL,ORDER=hfact job_flex.sh
    sbatch --export=ALL,ORDER=hfact,REQUIRE_NET=982.4 job_flex.sh

`--design-json {"label":,"places":[...],"x":[...]}` certifies a design from elsewhere
(e.g. a GA's winner) alongside PROCESS's.

## The answer, N = 262144

| hfact sigma | operable (run at all) | at rated power |
|---|---|---|
| 0.10 | 0.431 | 0.091 |
| 0.05 | 0.399 | 0.139 |
| 0.02 | 0.377 | 0.172 |

The machine is valid in every world -- the seven build constraints never fail -- so every
failure is an *operating* failure. It binds on c83, c35 and the lifted pack rule in ~100 %
of operable draws, all fixed by the build, so it has no margin the operator can trade, and
the knobs buy almost nothing (T_e alone 0.387, He alone 0.402, both 0.412, plus the
operating field 0.412). Note the two questions move in *opposite* directions with
uncertainty: without a power floor its upside tail rescues marginal worlds, with one the
downside tail drops below rated power and the upside is worth nothing.

**Implementation notes.** The Jacobian is taken only where SLSQP asks for one (value at
~23 points/draw, gradient at ~10), via separate caches and a fused
`jacfwd(..., has_aux=True)` trace -- 1.9x, bit-identical. One vector-valued `ineq`
replaces 13 dicts. `multiprocessing` with `spawn`, one compile per worker, JAX's
persistent compilation cache.
"""
from __future__ import annotations

import argparse, json, os, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
SCRATCH = HERE  # kept: --jit-cache defaults under it


def _parse(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=1024)
    p.add_argument("--which", default="deterministic", choices=("deterministic", "robust", "both"))
    p.add_argument("--sigma", type=float, default=0.10, help="hfact belief sigma")
    p.add_argument("--knobs", default="both", choices=("te", "he", "both"))
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--order", default="index", choices=("index", "hfact"),
                   help="the order draws are walked in (warm-start locality)")
    p.add_argument("--chunks-per-worker", type=int, default=4,
                   help="slices per worker; >1 balances the load dynamically "
                        "(inoperable draws skip phase 2, so contiguous slices are uneven)")
    p.add_argument("--restarts", type=int, default=0, choices=(0, 1, 2),
                   help="extra phase-2 starts (1 = the running incumbent, 2 = it and "
                        "the design's own set-points). 0 reproduces flex_orig exactly")
    p.add_argument("--out", default=None)
    p.add_argument("--jit-cache", default=os.path.join(SCRATCH, "jitcache"))
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--require-net", type=float, default=0.0,
                   help="net electric MW to require (c16, which is NOT in this "
                        "constraint set). Default 0 = off. Note the file's literal "
                        "1000.0 makes PROCESS's own design infeasible at the nominal: "
                        "the port delivers 982.4 MW there (~1.8%% low, a deliberate "
                        "divergence) and c16 is an equality the optimum sits exactly "
                        "on, so it has no margin to absorb that.")
    p.add_argument("--design-json", default=None,
                   help='a design to certify: {"label":, "places": [...], "x": [...]}')
    p.add_argument("--figures", action="store_true",
                   help="draw the figures from the runs already in out/ and stop")
    p.add_argument("--dsm", action="store_true",
                   help="render the graph this analysis solves and stop")
    return p.parse_args(argv)


# --------------------------------------------------------------------------- worker
_STATE = {}


def _setup(cfg):
    """Everything that is per-process: jax config, the model, the compiled programs."""
    import jax
    jax.config.update("jax_enable_x64", True)
    if cfg["jit_cache"]:
        os.makedirs(cfg["jit_cache"], exist_ok=True)
        jax.config.update("jax_compilation_cache_dir", cfg["jit_cache"])
        jax.config.update("jax_persistent_cache_min_entry_size_bytes", 0)
        jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.2)
    import dataclasses
    import numpy as np
    import jax.numpy as jnp
    from common import deterministic_values
    from functional_process.cottax.architectures import lift, ouu, session
    from functional_process.configurations import kinds

    knobs = {"te": (kinds.TE,),
             "he": (".physics.f_nd_alpha_thermal_electron",),
             "both": (kinds.TE, ".physics.f_nd_alpha_thermal_electron")}[cfg["knobs"]]
    live = session.open_session("stellarator_helias")
    dv, cv = deterministic_values(live, "one")
    beliefs = tuple(dataclasses.replace(b, a=cfg["sigma"]) if b.path == ".physics.hfact" else b
                    for b in kinds.BELIEFS)
    model = ouu.two_stage(live, n=cfg["n"], alpha=0.9, seed=0, lifts=(lift.lift_winding_pack,),
                          beliefs=beliefs,
                          held=tuple(kinds.BUILD_LEAVES) + tuple(kinds.ECONOMIC),
                          design_values=dv, closing_values=cv)
    places = [v.spelling for v in model.design]
    op = np.array([places.index(p) for p in knobs])
    fns = ouu.make(model, hoist=False)
    run, L = fns["run_batch"], fns["layout"]

    @jax.jit
    def row(z, xf, th, st):
        x = jnp.asarray(xf).at[jnp.asarray(op)].set(jnp.asarray(z))
        return run(x, th, st)[0]

    jrow = jax.jit(jax.jacfwd(row))

    @jax.jit
    def vjrow(z, xf, th, st):
        def twice(zz):
            out = row(zz, xf, th, st)
            return out, out
        J, y = jax.jacfwd(twice, has_aux=True)(jnp.asarray(z, float))
        return y, J

    _STATE.update(
        jax=jax, jnp=jnp, np=np, ouu=ouu, kinds=kinds, model=model, places=places,
        op=op, knobs=knobs, L=L, row=row, jrow=jrow, vjrow=vjrow,
        lo=np.asarray(model.lower, float), hi=np.asarray(model.upper, float),
        require_net=float(cfg.get("require_net", 0.0) or 0.0),
        names=[c.spelling.rsplit(".", 1)[-1] for c in model.constraints]
              + ([] if not cfg.get("require_net", 0.0) else ["c16_net"]))
    return _STATE


def _compile(state, x0):
    """Trace and compile the three programs once, on draw 0."""
    jax, np, jnp = state["jax"], state["np"], state["jnp"]
    th = jax.tree_util.tree_map(lambda a: a[0:1], state["model"].theta)
    st = jnp.asarray(state["model"].starts0[0:1])
    z = np.ascontiguousarray(np.asarray(x0, float)[state["op"]])
    jax.block_until_ready(state["row"](z, x0, th, st))
    jax.block_until_ready(state["jrow"](z, x0, th, st))
    jax.block_until_ready(state["vjrow"](z, x0, th, st))


def solve_sample(state, i, x_fixed, z0, extra_starts=()):
    """The operator's problem for draw `i`, in two phases. Identical mathematics to
    `flex_orig.solve_sample`; only *when* a Jacobian is computed differs.

    `extra_starts` are further points phase 2 is run from, the cheapest **feasible**
    answer winning. Phase 2 is a local solve of a non-convex problem and its answer
    depends sharply on where it starts: walking the draws in `hfact` order rather
    than index order moves the coe p95 of this problem from 795 to 310 without
    changing a single operability verdict. With no extra start this reduces exactly
    to `flex_orig`'s single solve from `z_psi`.
    """
    import numpy as np
    from scipy.optimize import minimize
    jax, jnp, ouu = state["jax"], state["jnp"], state["ouu"]
    model, op, L = state["model"], state["op"], state["L"]
    c_coe, c_net, c_conv, (g0, g1) = L["c_coe"], L["c_net"], L["c_conv"], L["c_g"]
    NG, nz = g1 - g0, len(op)
    FEAS = ouu.G_TOL
    th = jax.tree_util.tree_map(lambda a: a[i:i + 1], model.theta)
    st = jnp.asarray(model.starts0[i:i + 1])
    vc, jc = {}, {}

    def V(z):
        z = np.ascontiguousarray(np.asarray(z, float))
        k = z.tobytes()
        r = vc.get(k)
        if r is None:
            r = vc[k] = np.asarray(state["row"](z, x_fixed, th, st))
        return r

    def J(z):
        z = np.ascontiguousarray(np.asarray(z, float))
        k = z.tobytes()
        r = jc.get(k)
        if r is None:
            if k in vc:                       # the value is already paid for
                r = jc[k] = np.asarray(state["jrow"](z, x_fixed, th, st))
            else:                             # one fused trace gives both
                y, j = state["vjrow"](z, x_fixed, th, st)
                vc[k] = np.asarray(y)
                r = jc[k] = np.asarray(j)
        return r

    # `c16` -- net electric power >= what the plant is specified for -- is in the
    # file's `icc` but not in this constraint set (it is one of the two equalities,
    # and only `c2` is closed). Without it the operator may satisfy every remaining
    # limit by producing less, and a search over the design may "win" on coe purely by
    # building a bigger plant than the one specified. It is carried here as a
    # normalised inequality alongside the rest.
    req = state["require_net"]

    def G(z):
        """The constraint vector: the graph's, plus the power requirement."""
        r = V(z)
        g = r[g0:g1]
        return g if req <= 0 else np.append(g, (req - r[c_net]) / req)

    def GJ(z):
        j = J(z)[g0:g1]
        return j if req <= 0 else np.vstack([j, -J(z)[c_net] / req])

    ng = NG + (0 if req <= 0 else 1)
    bnds = list(zip(state["lo"][op], state["hi"][op], strict=True))

    # -- phase 1: psi = min_z max_j g_j, as `min t` s.t. `g_j(z) - t <= 0`
    w0 = np.concatenate([np.asarray(z0, float), [float(G(z0).max())]])
    slack = [{"type": "ineq",
              "fun": lambda w: w[-1] - G(w[:-1]),
              "jac": lambda w: np.hstack([-GJ(w[:-1]), np.ones((ng, 1))])}]
    p1 = minimize(lambda w: float(w[-1]), w0,
                  jac=lambda w: np.concatenate([np.zeros(nz), [1.0]]),
                  method="SLSQP", bounds=[*bnds, (-10.0, 10.0)], constraints=slack,
                  options={"maxiter": 80, "ftol": 1e-10})
    z_psi = np.ascontiguousarray(np.asarray(p1.x[:-1], float))
    psi = float(G(z_psi).max())
    # A world counts as operable only if the plant actually generates. `c16` (the net
    # electric lower limit) is not in this constraint set, so without the `net > 0`
    # test a draw can satisfy every remaining constraint by producing nothing: 41 of
    # 113075 operable draws at N=262144 had net <= 0 (down to -63.8 MW) and coe up to
    # 6.7e21, which poisons any unconditioned statistic.
    net_psi = float(V(z_psi)[c_net])
    operable = bool(V(z_psi)[c_conv] > 0.5 and psi <= FEAS and net_psi > 0.0)

    # -- phase 2: cheapest operating point among the feasible ones
    z_star, note = z_psi, "psi only"
    if operable:
        # `psi > 0` in every operable draw (c83 sits at +1.6e-4 and the operator cannot
        # move it), so `{g <= 0}` is EMPTY and `min coe s.t. g <= 0` has no solution.
        # SLSQP hides that -- it stops on its iteration limit and the row is then
        # accepted because `max g <= FEAS` -- and the coe it returns is not an optimum.
        # The reachable set is `g <= max(FEAS, psi)`, non-empty by construction.
        slack2 = max(FEAS, psi)
        cons = [{"type": "ineq", "fun": lambda z: slack2 - G(z),
                 "jac": lambda z: -GJ(z)}]
        best = float(V(z_psi)[c_coe])
        starts = [z_psi]
        for s in extra_starts:
            s = np.ascontiguousarray(np.asarray(s, float))
            if not any(np.array_equal(s, t) for t in starts):
                starts.append(s)
        for k, s in enumerate(starts):
            p2 = minimize(lambda z: float(V(z)[c_coe]), s,
                          jac=lambda z: np.asarray(J(z)[c_coe]),
                          method="SLSQP", bounds=bnds, constraints=cons,
                          options={"maxiter": 80, "ftol": 1e-9})
            pz = np.ascontiguousarray(np.asarray(p2.x, float))
            r2 = V(pz)
            if r2[c_conv] > 0.5 and r2[g0:g1].max() <= FEAS and r2[c_coe] < best:
                best, z_star = float(r2[c_coe]), pz
                note = "coe minimised" if k == 0 else f"coe minimised (start {k})"
    r = V(z_star)
    g = G(z_star)          # the graph's conditions plus the power requirement
    places, kinds, names = state["places"], state["kinds"], state["names"]
    knobs = state["knobs"]
    return {
        "i": int(i), "operable": operable, "psi": psi, "note": note,
        "converged": bool(r[c_conv] > 0.5),
        "feasible": bool(r[c_conv] > 0.5 and r[c_net] > 0 and g.max() <= FEAS),
        "coe": float(r[c_coe]), "net_mw": float(r[c_net]),
        "te": float(z_star[0]) if kinds.TE in knobs else float(x_fixed[places.index(kinds.TE)]),
        "f_alpha": (float(z_star[-1]) if ".physics.f_nd_alpha_thermal_electron" in knobs
                    else float(x_fixed[places.index(".physics.f_nd_alpha_thermal_electron")])),
        "max_g": float(g.max()),
        "active": [names[k] for k in range(len(names)) if g[k] > -1e-3],
        "calls": len(vc), "jcalls": len(jc),
    }


def _slice(args):
    """One worker: set up, compile, walk a contiguous slice warm-started along it."""
    cfg, idx, x_fixed, tag = args
    import numpy as np
    state = _STATE if _STATE else _setup(cfg)
    x_fixed = np.asarray(x_fixed, float)
    if "compiled" not in state:
        t = time.perf_counter()
        _compile(state, x_fixed)
        state["compiled"] = time.perf_counter() - t
    began = time.perf_counter()
    rows, z = [], np.ascontiguousarray(x_fixed[state["op"]])
    z_home = np.ascontiguousarray(x_fixed[state["op"]])
    extra = cfg.get("restarts", 0)
    for i in idx:
        more = ()
        if extra:
            more = (z,) if extra == 1 else (z, z_home)
        r = solve_sample(state, int(i), x_fixed, z, extra_starts=more)
        rows.append(r)
        if r["operable"]:
            kinds, knobs = state["kinds"], state["knobs"]
            z = np.ascontiguousarray(np.array(
                [r["te"] if o == kinds.TE else r["f_alpha"] for o in knobs], float))
    try:
        import resource
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    except Exception:
        rss = 0.0
    return {"tag": tag, "rows": rows, "wall": time.perf_counter() - began,
            "compile": state.get("compiled", 0.0), "pid": os.getpid(), "rss_mb": rss}


def _init(cfg):
    _setup(cfg)


# --------------------------------------------------------------------------- driver
def _order(model, n, how):
    import numpy as np
    if how == "hfact":
        j = model.var_of[".physics.hfact"]
        return np.argsort(np.asarray(model.theta[j]).reshape(-1)[:n])
    return np.arange(n)


def summarise(rows, model, n, label, out=print):
    import numpy as np
    from collections import Counter
    feas = np.array([r["operable"] for r in rows])
    psi = np.array([r["psi"] for r in rows])
    coe = np.array([r["coe"] for r in rows]); net = np.array([r["net_mw"] for r in rows])
    te = np.array([r["te"] for r in rows])
    out(f"   operable in {feas.mean():.3f} of draws ({feas.sum()}/{n})"
        f"   psi p50 {np.median(psi):+.4f}  p95 {np.percentile(psi,95):+.4f}")
    if feas.any():
        q = lambda v, p: np.percentile(v[feas], p)
        out(f"   coe    p5 {q(coe,5):7.2f}  p50 {q(coe,50):7.2f}  p95 {q(coe,95):7.2f}")
        out(f"   net MW p5 {q(net,5):7.1f}  p50 {q(net,50):7.1f}  p95 {q(net,95):7.1f}"
            f"   spread {q(net,95)/max(q(net,5),1e-9):.2f}x")
        out(f"   T_e    p5 {q(te,5):6.2f}  p50 {q(te,50):6.2f}  p95 {q(te,95):6.2f} keV")
        act = Counter(a for r in rows if r["feasible"] for a in r["active"])
        out("   active: " + ", ".join(f"{k} {v/feas.sum():.2f}" for k, v in act.most_common(5)))
    out(f"   {np.mean([r['calls'] for r in rows]):.1f} value pts/draw, "
        f"{np.mean([r['jcalls'] for r in rows]):.1f} jac pts/draw")
    theta_q = np.asarray(model.Theta)[:n]
    scored = []
    for j, b in enumerate(model.beliefs):
        u = theta_q[:, j]
        if u.std() < 1e-12 or feas.std() < 1e-12:
            continue
        scored.append((float(np.corrcoef(u, feas.astype(float))[0, 1]), b.path))
    out("\n   what drives operability (corr of the belief's quantile with operable):")
    for c, path in sorted(scored, key=lambda t: -abs(t[0]))[:8]:
        out(f"      {c:+.3f} {('+' if c > 0 else '-') * min(20, int(abs(c)*40)):20s} {path}")


def figures(pattern="flex_n262144_s{sigma}_hfact"):
    """The figures, from the condensed `.npz` + `_summary.json` in `out/`."""
    import numpy as np, matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sigmas = ("0.10", "0.05", "0.02")
    d = {s: np.load(f"{OUT}/{pattern.format(sigma=s)}.npz", allow_pickle=True) for s in sigmas}
    summ = {s: json.load(open(f"{OUT}/{pattern.format(sigma=s)}_summary.json")) for s in sigmas}
    main_, fixed = d["0.10"], {"c82", "c83", "c32", "c34", "c35", "c65", "wp_width_r_min"}
    # Apply the `net > 0` correction to runs written before it was part of the test.
    for s_ in sigmas:
        d[s_] = {k: d[s_][k] for k in d[s_].files}
        d[s_]["operable"] = d[s_]["operable"] & (d[s_]["net_mw"] > 0.0)
        summ[s_]["operable"] = float(d[s_]["operable"].mean())
    op = main_ = d["0.10"], None
    main_, op = d["0.10"], d["0.10"]["operable"]
    short = [str(x).rsplit(".", 1)[-1] for x in main_["constraint_names"]]
    fig, ax = plt.subplots(2, 3, figsize=(16, 8.5))
    fig.suptitle("PROCESS's deterministic stellarator optimum under uncertainty -- "
                 f"N = {len(op):,} belief draws, operator re-optimises the set-points", fontsize=12)
    a = ax[0, 0]
    for s, c in zip(sigmas, ("tab:red", "tab:orange", "tab:brown")):
        a.hist(np.clip(d[s]["psi"], -0.1, 1.2), bins=120, histtype="step", lw=1.6, color=c,
               label=f"hfact $\\sigma$={s}  (operable {summ[s]['operable']:.3f})")
    a.axvline(0, color="k", lw=1); a.legend(fontsize=8)
    a.set_xlabel(r"$\psi$ = min$_z$ max$_j$ $g_j$")
    a.set_title(r"feasibility function $\psi$   ($\psi\leq0$ = operable)")
    for axis, key, colour, title, xlab in (
            (ax[0, 1], "net_mw", "tab:blue", "net electric power | operable", "MW"),
            (ax[0, 2], "coe", "tab:green", "cost of electricity | operable", "coe"),
            (ax[1, 0], "te", "tab:purple", r"operator's chosen $T_e$ | operable", "keV")):
        v = main_[key][op]
        axis.hist(np.clip(v, 0, 500) if key == "coe" else v, bins=80, color=colour, alpha=.85)
        axis.set_xlabel(xlab); axis.set_title(title)
    a = ax[1, 1]; psi = main_["psi"]
    a.hist(psi[op], bins=60, alpha=.75, color="tab:blue", label="operable")
    a.hist(np.clip(psi[~op], -0.1, 1.2), bins=60, alpha=.75, color="tab:red", label="not")
    a.set_yscale("log"); a.set_xlabel(r"$\psi$"); a.legend(fontsize=8)
    a.set_title(r"$\psi$ split by outcome")
    a = ax[1, 2]
    frac = np.array([main_["active"][op, j].mean() for j in range(len(short))])
    o = np.argsort(frac)
    a.barh([short[j] for j in o], frac[o],
           color=["tab:red" if short[j] in fixed else "tab:blue" for j in o])
    a.set_xlabel("fraction of operable draws where active")
    a.set_title("red = fixed by the build, blue = the operator can move it")
    plt.tight_layout(rect=(0, 0, 1, 0.96))
    plt.savefig(f"{OUT}/flex_histograms.png", dpi=140)
    print(f"wrote {OUT}/flex_histograms.png")


def main(argv=None):
    a = _parse(argv)
    if a.figures:
        return figures()
    if a.dsm:
        from flexibility_dsm import render  # noqa: PLC0415
        return render()
    cfg = {"n": a.n, "sigma": a.sigma, "knobs": a.knobs, "jit_cache": a.jit_cache,
           "require_net": a.require_net,
           "restarts": a.restarts}
    t0 = time.perf_counter()
    state = _setup(cfg)
    np, model, places = state["np"], state["model"], state["places"]
    print(f"setup {time.perf_counter()-t0:.1f} s   hfact sigma {a.sigma}   knobs {a.knobs} -> "
          f"{[o.rsplit('.',1)[-1] for o in state['knobs']]}", flush=True)

    designs = {}
    if a.which in ("deterministic", "both"):
        designs["deterministic"] = np.asarray(model.x0, float)
    if a.design_json:
        # A design from elsewhere (e.g. the GA's winner): {"places": [...], "x": [...]}
        spec = json.load(open(a.design_json))
        x = np.asarray(model.x0, float).copy()
        for k, v in zip(spec["places"], spec["x"], strict=True):
            x[places.index(k)] = v
        designs[spec.get("label", "candidate")] = x
    if a.which in ("robust", "both"):
        x = np.asarray(model.x0, float).copy()
        robust = "/home/tbogaarts/PROCESS/paper_tests/out/ouu9_alpha0.9_te_design.json"
        for k, v in json.load(open(robust))["robust"]["x"].items():
            if k in places:
                x[places.index(k)] = v
        designs["robust"] = x

    order = _order(model, a.n, a.order)
    out = {}
    timings = {}
    for label, x_fixed in designs.items():
        build = [i for i in range(len(places)) if i not in state["op"]]
        print(f"\n=== {label}: " + ", ".join(
            f"{places[i].rsplit('.',1)[-1]} {x_fixed[i]:.4g}" for i in build), flush=True)
        began = time.perf_counter()
        if a.workers <= 1:
            res = [_slice((cfg, order, x_fixed.tolist(), 0))]
        else:
            import multiprocessing as mp
            nchunk = max(1, a.workers * max(1, a.chunks_per_worker))
            chunks = [c for c in np.array_split(order, nchunk) if len(c)]
            ctx = mp.get_context("spawn")
            with ctx.Pool(a.workers, initializer=_init, initargs=(cfg,)) as pool:
                res = list(pool.imap_unordered(
                    _slice, [(cfg, c, x_fixed.tolist(), k) for k, c in enumerate(chunks)]))
        wall = time.perf_counter() - began
        rows = sorted((r for s in res for r in s["rows"]), key=lambda r: r["i"])
        by_pid = {}
        for sres in res:
            by_pid[sres["pid"]] = by_pid.get(sres["pid"], 0.0) + sres["wall"]
        timings[label] = {"wall": wall, "workers": a.workers,
                          "chunks": len(res),
                          "slice_wall": [s["wall"] for s in res],
                          "worker_wall": sorted(by_pid.values()),
                          "rss_mb": [s.get("rss_mb", 0.0) for s in res],
                          "slice_compile": [s["compile"] for s in res]}
        summarise(rows, model, a.n, label)
        ww = timings[label]["worker_wall"]
        print(f"   {wall:.1f} s total ({a.workers} workers, {len(res)} chunks); "
              f"busy per worker {min(ww):.1f}-{max(ww):.1f} s, "
              f"compile {max(s['compile'] for s in res):.1f} s, "
              f"peak RSS/worker {max(timings[label]['rss_mb']):.0f} MB", flush=True)
        out[label] = rows

    path = a.out or f"{OUT}/flex_n{a.n}_s{a.sigma:.2f}_{a.knobs}.json"
    json.dump({"n": a.n, "places": places, "operating": list(state["knobs"]),
               "workers": a.workers, "order": a.order, "restarts": a.restarts,
               "sigma": a.sigma, "knobs": a.knobs, "timings": timings,
               "designs": {k: v.tolist() for k, v in designs.items()}, "rows": out},
              open(path, "w"), indent=1)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
