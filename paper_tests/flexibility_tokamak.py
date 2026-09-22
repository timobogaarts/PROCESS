"""Flexibility analysis of PROCESS's driven tokamak (`large_tokamak_nof`, the EU-DEMO
shape): fix the machine, let the operator re-optimise per belief draw.

The tokamak counterpart of `flexibility.py` (the stellarator study), same question,
same two phases per draw, same statistics; what differs is the machine and what the
graph had to say about it -- `functional_process/cottax/architectures/driven.py` is
the recipe, `configurations/kinds_tokamak.py` the sort, and
`paper_tests/decision_kinds_tokamak.md` the census.

**The machine.** PROCESS's DEMO is *driven*, not ignited: 75 MW of ECRH heating during
the burn plus current-drive power for a 44 % non-inductive fraction, capped at 200 MW
(`c30`), with `hfact` (H98y2) the iteration variable that closes the power balance
`c2` -- and PROCESS's converged design holds it at its **upper bound, 1.2**. The figure
of merit is the major radius, which sits on its lower bound (8 m), so the operating
point PROCESS reports is one feasible point, not a cost optimum.

**The formulation.**

- *Beliefs*, two groups (`kinds_tokamak.BELIEFS`): PLASMA -- `hfact` lognormal
  (sigma 0.10, `--sigma`), `alphan` / `alphat` +-20 %, `T_i/T_e` U[0.85, 1],
  `f_p_alpha_plasma_deposited` U[0.90, 0.99], the tungsten fraction +-20 %; LIMITS --
  the beta limit's coefficient (`f_beta_norm_max`, +-20 % on Wesson's `4 l_i`, a
  `Redefine` of the Wesson node: on the tokamak branch the threshold is computed, the
  file's `beta_norm_max = 3.0` is dead) and `f_t_alpha_energy_confinement_min` U[3, 6].
- *Build*, fixed at PROCESS's converged design: the twelve build `ixc`, `q95` and
  `f_c_plasma_non_inductive` (operating by kind, held: the CS and PF are sized to the
  current), and the **installed H&CD power**, a new boundary input the cost node reads
  in place of the used power and `c30` reads in place of `p_hcd_injected_max` (two
  `Rewire`s: PROCESS costs the used power and caps it by a number nothing pays for). At
  the deterministic point the installed power is the used 75 MW plus the current-drive
  power, zero margin.
- *Closures per world* (`closing.close_conditions`, three root finds, two of them on
  one cycle and flattened into one square problem of five unknowns): `c2` by the
  heating power (`hfact` is a belief now; at the operator's (n, T) the balance says
  what P_aux must be), `c62` -- an inequality held with equality, the helium particle
  balance n_He = rho* tau_E S_alpha -- by the thermal alpha fraction, `c1` by the beta
  (PROCESS's `ixc 5` is the copy of a computed quantity).
- *Operator knobs*: the density (under Greenwald, `c5`), `T_e`, the xenon fraction.
- *Frozen build outputs* (`driven.FREEZES`): the stage split found the whole PF coil
  system re-sized per world (the equilibrium currents read the poloidal beta) and
  with it the cryostat, the structure masses and the buildings; one `Cut` of the
  coils' rated currents (`^built.pf_coil.c_pf_cs_coils_peak_ma`) with a capacity
  inequality `|I_world| <= |I_built|` per coil, plus exact freezes of geometry a node
  happens to own beside a per-world quantity, put them back in the first stage.
- *Per draw*: phase 1 `psi = min_z max_j g_j(z)` over the conditions the operator can
  move (the eight build constraints -- `c25`, `c31`-`c36`, `c65` -- are constants given
  the build and are reported once, not posed: cottax refuses a constant as a
  condition), then phase 2 `min coe` s.t. `g <= max(G_TOL, psi)`, as `flexibility.py`.
  `c16` (net electric >= 400 MW) is an inequality of the file and is in the set;
  `--drop-c16` asks "can it be run at all" instead.

## Recipe

    PY=~/miniconda3/envs/process_port/bin/python   # or ~/miniconda/envs/process_port
    SC=<scratch>; mkdir -p $SC/cottax_head && git -C ~/jaxgraph archive 6488a1c src | tar -x -C $SC/cottax_head
    cd ~/PROCESS
    export JAX_PLATFORMS=cpu PYTHONPATH=~/PROCESS:$SC/cottax_head/src:~/PROCESS/paper_tests
    $PY paper_tests/flexibility_tokamak.py --n 1024 --workers 8 --order hfact
    $PY paper_tests/flexibility_tokamak.py --figures
    cd paper_tests/cluster && sbatch --export=ALL,ORDER=hfact job_flex_tok.sh

## The answer, N = 262144 (Snellius, 40 workers, ~10-18 min a sigma; `out/flextok_*`)

| hfact sigma | operable (net >= 400 MW) | coe p5 / p50 / p95 | P_heat at zero | cheaper than the nominal |
|---|---|---|---|---|
| 0.10 | 0.729 | 315 / 380 / 464 | 0.94 | 0.46 |
| 0.05 | 0.863 | 320 / 385 / 460 | 0.98 | 0.41 |
| 0.02 | 0.952 | 335 / 389 / 448 | 0.99 | 0.34 |
| 0.10, `--drop-c16` (N = 65536) | 0.989 | 320 / 413 / 779 | 0.93 | 0.33 |

**The reference line is not PROCESS's 510.** PROCESS's reported point is one feasible
point of an 8 m machine (the figure of merit sits on its bound) with `c62` slack (rho* =
7.1 against the 5 required) and 75 MW of heating it does not need: the operator, in the
nominal world, holds `c62` with equality (f_He 0.068 against 0.086), turns the heating
off (P_heat = 0; the balance would ask for *negative* heating, so its lower bound is
active) and runs at 530 MW net for coe **374**, 0.73 of PROCESS's. Every operable world
is cheaper than PROCESS's reported coe; 46 % are cheaper than the operator's nominal.

**What the build buys and what it does not.** The CS is used to capacity in every
operable world (`c60`, `c72` and the PF/CS coil-current capacity active in 100 %; PROCESS's
own optimum sits on `c60` / `c72` too), the divertor guideline `c68` in 82 % and the
Greenwald cap `c5` in 73 %: the operator has no margin to trade on the pulse, and what the
knobs buy is the choice of (n, T) inside those limits. The 84.8 MW installed is idle but
for the 9.8 MW of current drive: no operable world at any sigma uses more than 75 MW of
heating, and 94 % use none. Inoperable worlds (27 % at sigma 0.10) fail on the divertor
guideline `c68` (38 %), the 400 MW requirement `c16` (37 %: bad confinement), the heating
power's lower bound (16 %: the balance needs negative heating everywhere the other limits
allow -- over-performance the model cannot run steadily) and `c5` (5 %); `hfact` is the
only belief that matters (corr 0.66 with operable; `T_i/T_e` 0.17, rho* -0.14, the beta
factor and the tungsten fraction 0.00). Without the 400 MW requirement the machine runs in
99 % of worlds, and the 1 % that fail hit the installed-power cap `c30`.

**Implementation notes.** 0.12-0.15 s a draw per core after a ~45 s compile; 3.9 GB peak
RSS per worker on rome (64 workers were OOM-killed, 40 fit). The Jacobian is guarded: at a
point where a nested solve fails, optimistix's implicit adjoint hands lineax a
non-finite matrix and equinox raises through the callback -- the value survives (its
`Converged` says so), so a raised Jacobian is replaced by zeros and SLSQP sees the point
as infeasible. 53 of 191208 operable rows at sigma 0.10 have a non-finite coe and are
left out of the quantiles.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import time

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
PREFIX = "flextok"
NAME = "large_tokamak_nof"


def _parse(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=1024)
    p.add_argument("--sigma", type=float, default=0.10, help="hfact belief sigma")
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--order", default="hfact", choices=("index", "hfact"),
                   help="the order draws are walked in (warm-start locality)")
    p.add_argument("--chunks-per-worker", type=int, default=4)
    p.add_argument("--out", default=None)
    p.add_argument("--jit-cache", default=os.path.join(HERE, "jitcache"))
    p.add_argument("--drop-c16", action="store_true",
                   help="leave the net electric requirement (c16, 400 MW) out: "
                        "'can it be run at all' rather than 'at rated power'")
    p.add_argument("--no-freeze", action="store_true",
                   help="skip driven.FREEZES (the build re-sized per world; for the split's appendix)")
    p.add_argument("--figures", action="store_true",
                   help="draw the figures from the runs already in out/ and stop")
    p.add_argument("--dsm", action="store_true",
                   help="render the graph this analysis solves as DSM pages (out/dsm/large_tokamak_nof/) and stop")
    p.add_argument("--point", action="store_true",
                   help="print the deterministic point under the new closures against PROCESS's and stop")
    return p.parse_args(argv)


# --------------------------------------------------------------------------- worker
_STATE = {}


def deterministic_values(live, proc, pairings):
    """PROCESS's converged design as `driven.two_stage` takes it: `design_values`
    (`ixc` order minus the closing variables) and `closing_values`; the heating
    power, which is not an `ixc`, starts at the file's 75 MW.
    """
    from functional_process.cottax.architectures.sand import iteration_variable_path
    design = [iteration_variable_path(i).spelling for i in live.reference.ixc]
    closing = set(pairings.values())
    dv = [proc["x"][v] for v in design if v not in closing]
    cv = {v: proc["x"].get(v, 75.0) for v in closing}
    return dv, cv


def _setup(cfg):
    """Everything that is per-process: jax config, the model, the compiled programs."""
    import jax
    jax.config.update("jax_enable_x64", True)
    if cfg["jit_cache"]:
        pathlib.Path(cfg["jit_cache"]).mkdir(exist_ok=True, parents=True)
        jax.config.update("jax_compilation_cache_dir", cfg["jit_cache"])
        jax.config.update("jax_persistent_cache_min_entry_size_bytes", 0)
        jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.2)
    import dataclasses

    import jax.numpy as jnp
    import numpy as np
    from common import process_reference

    from functional_process.configurations import kinds_tokamak as kinds
    from functional_process.cottax.architectures import driven, ouu, session

    live = session.open_session(NAME)
    proc = process_reference(NAME)
    dv, cv = deterministic_values(live, proc, driven.PAIRINGS)
    # The installed power: what PROCESS's own point uses, zero margin.
    dv2, cv2 = deterministic_values(live, proc, driven.PROCESS_PAIRINGS)
    at_process = driven.process_point(live, dv2, cv2)
    installed = at_process["installed"]
    beliefs = tuple(dataclasses.replace(b, a=cfg["sigma"]) if b.path == ".physics.hfact" else b
                    for b in kinds.BELIEFS)
    model = driven.two_stage(live, n=cfg["n"], alpha=0.9, seed_=0, beliefs=beliefs,
                             design_values=dv, closing_values=cv, installed=installed,
                             freezes=() if cfg.get("no_freeze") else driven.FREEZES)
    places = [v.spelling for v in model.design]
    knobs = kinds.KNOBS
    op = np.array([places.index(p) for p in knobs])
    # Which conditions the operator can move: the ones whose node varies per world.
    varying = {n.spelling for n in model.stages.varying}
    owner = {v.spelling: n.spelling for v, n in model.closed.graph.graph.owners.items()}
    movable, fixed = [], []
    for k, c in enumerate(model.constraints):
        if isinstance(c, ouu.RecourseBound) or owner.get(c.spelling) in varying:
            movable.append(k)
        else:
            fixed.append(k)
    names = [c.spelling.rsplit(".", 1)[-1] for c in model.constraints]
    if cfg.get("drop_c16"):
        movable = [k for k in movable if names[k] != "c16"]
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
        jax=jax, jnp=jnp, np=np, ouu=ouu, kinds=kinds, driven=driven, model=model,
        places=places, op=op, knobs=knobs, L=L, row=row, jrow=jrow, vjrow=vjrow,
        lo=np.asarray(model.lower, float), hi=np.asarray(model.upper, float),
        names=names, movable=np.array(movable, int), fixed=np.array(fixed, int),
        mov_names=[names[k] for k in movable],
        scale=np.abs(np.asarray(model.x0, float)[op]),
        proc=proc, at_process=at_process, installed=installed,
        i_heating=model.unknowns.index(driven.HEATING),
        i_he=[u.spelling for u in model.unknowns].index(".physics.f_nd_alpha_thermal_electron"),
        i_beta=[u.spelling for u in model.unknowns].index(".physics.beta_total_vol_avg"),
    )
    return _STATE


def _compile(state, x0):
    jax, np, jnp = state["jax"], state["np"], state["jnp"]
    th = jax.tree_util.tree_map(lambda a: a[0:1], state["model"].theta)
    st = jnp.asarray(state["model"].starts0[0:1])
    z = np.ascontiguousarray(np.asarray(x0, float)[state["op"]])
    jax.block_until_ready(state["row"](z, x0, th, st))
    jax.block_until_ready(state["jrow"](z, x0, th, st))
    jax.block_until_ready(state["vjrow"](z, x0, th, st))


def solve_sample(state, i, x_fixed, z0):
    """The operator's problem for draw `i`, in two phases (`flexibility.solve_sample`,
    over the movable conditions only).
    """
    import numpy as np
    from scipy.optimize import minimize
    jax, jnp, ouu = state["jax"], state["jnp"], state["ouu"]
    model, op, L = state["model"], state["op"], state["L"]
    c_coe, c_net, c_conv, (g0, g1) = L["c_coe"], L["c_net"], L["c_conv"], L["c_g"]
    c_u0, c_u1 = L["c_u"]
    mov, nz = state["movable"], len(op)
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
            # The forward row survives a point where a nested solve fails (its
            # `Converged` says so); the Jacobian does not -- optimistix's implicit
            # adjoint hands lineax a non-finite matrix and equinox raises through the
            # callback (task 27017074_0 on Snellius died of it, 2026-09-22). A zero
            # row there is what SLSQP should see: the point is infeasible outright.
            try:
                if k in vc:
                    r = np.asarray(state["jrow"](z, x_fixed, th, st))
                else:
                    y, j = state["vjrow"](z, x_fixed, th, st)
                    vc[k] = np.asarray(y)
                    r = np.asarray(j)
            except Exception:  # noqa: BLE001 -- equinox's runtime error, whatever it wraps
                r = np.zeros((len(V(z)), len(z)))
            r = jc[k] = np.where(np.isfinite(r), r, 0.0)
        return r

    def G(z):
        r = V(z)
        g = r[g0:g1][mov]
        # A row whose closure did not converge is infeasible outright.
        return np.where(np.isfinite(g), g, ouu.FAILED_G) if r[c_conv] > 0.5 else np.full_like(g, ouu.FAILED_G)

    def GJ(z):
        j = J(z)[g0:g1][mov]
        return np.where(np.isfinite(j), j, 0.0)

    ng = len(mov)
    # SLSQP works in the knobs' own units and starts from a unit Hessian: a density of
    # 8e19 m^-3 beside a temperature of 12 keV and a xenon fraction of 6e-4 would never
    # move. Every knob is scaled by its nominal value, so the optimiser sees O(1).
    scale = state["scale"]
    bnds = list(zip(state["lo"][op] / scale, state["hi"][op] / scale, strict=True))
    Gs, GJs = (lambda s: G(s * scale)), (lambda s: GJ(s * scale) * scale[None, :])
    Vs, Js = (lambda s: V(s * scale)), (lambda s: J(s * scale) * scale[None, :])

    s0 = np.asarray(z0, float) / scale
    w0 = np.concatenate([s0, [float(Gs(s0).max())]])
    slack = [{"type": "ineq",
              "fun": lambda w: w[-1] - Gs(w[:-1]),
              "jac": lambda w: np.hstack([-GJs(w[:-1]), np.ones((ng, 1))])}]
    p1 = minimize(lambda w: float(w[-1]), w0,
                  jac=lambda w: np.concatenate([np.zeros(nz), [1.0]]),
                  method="SLSQP", bounds=[*bnds, (-10.0, 10.0)], constraints=slack,
                  options={"maxiter": 100, "ftol": 1e-10})
    s_psi = np.ascontiguousarray(np.asarray(p1.x[:-1], float))
    g_psi = Gs(s_psi)
    psi = float(g_psi.max())
    binding = state["mov_names"][int(np.argmax(g_psi))]
    net_psi = float(Vs(s_psi)[c_net])
    operable = bool(Vs(s_psi)[c_conv] > 0.5 and psi <= FEAS and net_psi > 0.0)

    s_star, note = s_psi, "psi only"
    if operable:
        slack2 = max(FEAS, psi)
        cons = [{"type": "ineq", "fun": lambda s: slack2 - Gs(s), "jac": lambda s: -GJs(s)}]
        best = float(Vs(s_psi)[c_coe])
        p2 = minimize(lambda s: float(Vs(s)[c_coe]), s_psi,
                      jac=lambda s: np.asarray(Js(s)[c_coe]),
                      method="SLSQP", bounds=bnds, constraints=cons,
                      options={"maxiter": 100, "ftol": 1e-9})
        ps = np.ascontiguousarray(np.asarray(p2.x, float))
        r2 = Vs(ps)
        # SLSQP's answer sits a hair outside its own constraints; a normalised 1e-6 is
        # noise against `G_TOL` (2e-3) and the `psi` slack.
        if r2[c_conv] > 0.5 and Gs(ps).max() <= slack2 + 1e-6 and r2[c_coe] < best:
            best, s_star, note = float(r2[c_coe]), ps, "coe minimised"
        else:
            note = (f"psi only (phase 2: conv {int(r2[c_conv] > 0.5)}, max g {Gs(ps).max():+.1e} "
                    f"vs {slack2:+.1e}, coe {r2[c_coe]:.2f} vs {best:.2f}, {p2.message})")
    z_star = s_star * scale
    r = V(z_star)
    g = G(z_star)
    names, mov_names = state["names"], [state["names"][k] for k in mov]
    u = r[c_u0:c_u1]
    return {
        "i": int(i), "operable": operable, "psi": psi, "binding": binding, "note": note,
        "converged": bool(r[c_conv] > 0.5),
        "feasible": bool(r[c_conv] > 0.5 and r[c_net] > 0 and g.max() <= FEAS),
        "coe": float(r[c_coe]), "net_mw": float(r[c_net]),
        "ne": float(z_star[0]), "te": float(z_star[1]), "xe": float(z_star[2]),
        "p_heat": float(u[state["i_heating"]]), "f_he": float(u[state["i_he"]]),
        "beta": float(u[state["i_beta"]]),
        "max_g": float(g.max()),
        "g": [float(v) for v in g],
        "active": [mov_names[k] for k in range(len(mov_names)) if g[k] > -1e-3],
        "calls": len(vc), "jcalls": len(jc),
    }


def _slice(args):
    cfg, idx, x_fixed, tag = args
    import numpy as np
    state = _STATE or _setup(cfg)
    x_fixed = np.asarray(x_fixed, float)
    if "compiled" not in state:
        t = time.perf_counter()
        _compile(state, x_fixed)
        state["compiled"] = time.perf_counter() - t
    began = time.perf_counter()
    rows, z = [], np.ascontiguousarray(x_fixed[state["op"]])
    for i in idx:
        r = solve_sample(state, int(i), x_fixed, z)
        rows.append(r)
        if r["operable"]:
            z = np.array([r["ne"], r["te"], r["xe"]], float)
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


def fixed_report(state, x_fixed):
    """The build constraints -- constants given the build -- at the design, once."""
    import numpy as np
    z = np.ascontiguousarray(x_fixed[state["op"]])
    th = state["jax"].tree_util.tree_map(lambda a: a[-1:], state["model"].theta)
    st = state["jnp"].asarray(state["model"].starts0[-1:])
    r = np.asarray(state["row"](z, x_fixed, th, st))
    g0, _ = state["L"]["c_g"]
    return {state["names"][k]: float(r[g0 + k]) for k in state["fixed"]}


def summarise(rows, state, n, out=print, coe_nominal=None):
    from collections import Counter

    import numpy as np
    model, proc = state["model"], state["proc"]
    feas = np.array([r["operable"] for r in rows])
    psi = np.array([r["psi"] for r in rows])
    coe = np.array([r["coe"] for r in rows]); net = np.array([r["net_mw"] for r in rows])
    ph = np.array([r["p_heat"] for r in rows]); fhe = np.array([r["f_he"] for r in rows])
    te = np.array([r["te"] for r in rows]); ne = np.array([r["ne"] for r in rows]); xe = np.array([r["xe"] for r in rows])
    coe_process = proc["outputs"][".costs.coe"]
    out(f"   operable in {feas.mean():.3f} of draws ({feas.sum()}/{n})"
        f"   psi p50 {np.median(psi):+.4f}  p95 {np.percentile(psi, 95):+.4f}")
    summary = {"operable": float(feas.mean()), "n": int(n), "coe_process": coe_process,
               "non_finite_operable": int((feas & ~np.isfinite(coe)).sum()),
               "installed_mw": state["installed"], "heating_process_mw": state["at_process"]["heating"],
               "cd_process_mw": state["at_process"]["cd"]}
    finite = np.isfinite(coe) & np.isfinite(net)
    if (feas & ~finite).any():
        out(f"   {int((feas & ~finite).sum())} operable draws with a non-finite coe or net "
            f"(left out of the quantiles)")
    feas = feas & finite
    if feas.any():
        q = lambda v, p: float(np.percentile(v[feas], p))
        out(f"   coe    p5 {q(coe, 5):7.2f}  p50 {q(coe, 50):7.2f}  p95 {q(coe, 95):7.2f}"
            f"   (PROCESS's reported {coe_process:.2f}; p50/PROCESS {q(coe, 50) / coe_process:.3f}"
            + (f"; the operator's nominal {coe_nominal:.2f}, p50/nominal {q(coe, 50) / coe_nominal:.3f}, "
               f"cheaper than nominal in {np.mean(coe[feas] < coe_nominal):.2f}" if coe_nominal else "") + ")")
        out(f"   net MW p5 {q(net, 5):7.1f}  p50 {q(net, 50):7.1f}  p95 {q(net, 95):7.1f}")
        out(f"   P_heat p5 {q(ph, 5):7.1f}  p50 {q(ph, 50):7.1f}  p95 {q(ph, 95):7.1f} MW"
            f"   (file 75; installed {state['installed']:.1f}; cap 200)")
        out(f"   f_He   p5 {q(fhe, 5):7.4f}  p50 {q(fhe, 50):7.4f}  p95 {q(fhe, 95):7.4f}")
        out(f"   n_e    p5 {q(ne, 5):7.3g}  p50 {q(ne, 50):7.3g}  p95 {q(ne, 95):7.3g}")
        out(f"   T_e    p5 {q(te, 5):6.2f}  p50 {q(te, 50):6.2f}  p95 {q(te, 95):6.2f} keV")
        out(f"   Xe     p5 {q(xe, 5):7.2e}  p50 {q(xe, 50):7.2e}  p95 {q(xe, 95):7.2e}")
        act = Counter(a for r, ok in zip(rows, feas, strict=True) if ok for a in r["active"])
        out("   active: " + ", ".join(f"{k} {v / feas.sum():.2f}" for k, v in act.most_common(8)))
        summary.update({
            "coe_q": [q(coe, 5), q(coe, 50), q(coe, 95)],
            "net_q": [q(net, 5), q(net, 50), q(net, 95)],
            "p_heat_q": [q(ph, 5), q(ph, 50), q(ph, 95)],
            "f_he_q": [q(fhe, 5), q(fhe, 50), q(fhe, 95)],
            "te_q": [q(te, 5), q(te, 50), q(te, 95)],
            "ne_q": [q(ne, 5), q(ne, 50), q(ne, 95)],
            "xe_q": [q(xe, 5), q(xe, 50), q(xe, 95)],
            "active": {k: v / float(feas.sum()) for k, v in act.items()},
            "below_process_coe": float(np.mean(coe[feas] < coe_process)),
            "coe_nominal_operator": coe_nominal,
            "below_nominal_coe": None if not coe_nominal else float(np.mean(coe[feas] < coe_nominal)),
            "p_heat_zero": float(np.mean(ph[feas] < 0.5)),
            "p_heat_above_installed": float(np.mean(ph[feas] + state["at_process"]["cd"] > state["installed"])),
        })
    bind = Counter(r["binding"] for r in rows if not r["operable"])
    if bind:
        out("   what makes a world inoperable (the constraint psi sits on): " + ", ".join(
            f"{k} {v / (~feas).sum():.2f}" for k, v in bind.most_common(6)))
        summary["binding"] = {k: v / float((~feas).sum()) for k, v in bind.items()}
    out(f"   {np.mean([r['calls'] for r in rows]):.1f} value pts/draw, "
        f"{np.mean([r['jcalls'] for r in rows]):.1f} jac pts/draw")
    theta_q = np.asarray(model.Theta)[:n]
    scored = []
    for j, b in enumerate(model.beliefs):
        u = theta_q[:, j]
        if u.std() < 1e-12 or feas.std() < 1e-12:
            continue
        scored.append((float(np.corrcoef(u, feas.astype(float))[0, 1]), b.path))
    out("\n   what drives operability (corr of the belief's coordinate with operable):")
    for c, path in sorted(scored, key=lambda t: -abs(t[0])):
        out(f"      {c:+.3f} {('+' if c > 0 else '-') * min(20, int(abs(c) * 40)):20s} {path}")
    summary["operability_corr"] = {p: c for c, p in scored}
    return summary


def point_report(state, out=print):
    """The deterministic point under the new closures against PROCESS's."""
    import numpy as np
    model, proc, ap = state["model"], state["proc"], state["at_process"]
    y = np.asarray(state["row"](np.asarray(model.x0)[state["op"]], np.asarray(model.x0),
                                state["jax"].tree_util.tree_map(lambda a: a[-1:], model.theta),
                                state["jnp"].asarray(model.starts0[-1:])))
    L = state["L"]
    u = y[L["c_u"][0]:L["c_u"][1]]
    out("   PROCESS's own point (the port, c2 by the heating power, c1 by the beta, f_He at PROCESS's):")
    out(f"      heating {ap['heating']:.2f} MW (file 75), CD {ap['cd']:.2f} MW, installed = {ap['installed']:.2f} MW")
    for k in ("^cond.constraints.c62", ".physics.f_t_alpha_energy_confinement", ".costs.coe",
              ".heat_transport.p_plant_electric_net_mw", ".physics.p_fusion_total_mw"):
        v = ap["out"].get(k)
        if v is not None:
            out(f"      {k:50s} {float(np.asarray(v)):.6g}")
    out(f"      PROCESS itself: coe {proc['outputs']['.costs.coe']:.2f}, net {proc['outputs']['.heat_transport.p_plant_electric_net_mw']:.1f}, "
        f"P_fus {proc['outputs']['.physics.p_fusion_total_mw']:.1f}")
    out("   the same design under the study's closures (c62 held with equality at rho* = 5):")
    out(f"      heating {u[state['i_heating']]:.2f} MW, f_He {u[state['i_he']]:.4f} (PROCESS {proc['x']['.physics.f_nd_alpha_thermal_electron']:.4f}), "
        f"beta {u[state['i_beta']]:.5f} (PROCESS {proc['x']['.physics.beta_total_vol_avg']:.5f})")
    out(f"      coe {y[L['c_coe']]:.2f}, net {y[L['c_net']]:.1f} MW")
    g0, _ = L["c_g"]
    out("      conditions (normalised, <= 0 satisfied): " + ", ".join(
        f"{state['names'][k]} {y[g0 + k]:+.2e}" for k in range(len(state["names"]))))
    violated = {k: v for k, v in ap["conditions"].items() if v > 0 and "objf" not in k}
    out(f"      violated at PROCESS's own point in the port: {violated}")


def figures(pattern=PREFIX + "_n{n}_s{sigma}_hfact"):
    """The figures, from the run JSONs in `out/`."""
    import glob

    import matplotlib
    import numpy as np
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    files = sorted(glob.glob(f"{OUT}/{PREFIX}_n*_s*_hfact.json"))
    if not files:
        print("no runs in", OUT); return
    runs = {}
    for f in files:
        d = json.load(open(f))
        runs[d["n"], d["sigma"]] = d
    (n, sigma), main_ = max(runs.items(), key=lambda kv: (kv[0][0], kv[0][1]))  # largest N, sigma 0.10
    rows = main_["rows"]
    op = np.array([r["operable"] for r in rows])
    coe = np.array([r["coe"] for r in rows]); net = np.array([r["net_mw"] for r in rows])
    ph = np.array([r["p_heat"] for r in rows]); psi = np.array([r["psi"] for r in rows])
    coe_p = main_["summary"]["coe_process"]; inst = main_["summary"]["installed_mw"]
    fig, ax = plt.subplots(2, 3, figsize=(16, 8.5))
    fig.suptitle(f"PROCESS's deterministic tokamak (large_tokamak_nof) under uncertainty -- "
                 f"N = {n:,} belief draws, operator re-optimises n, T, Xe", fontsize=12)
    a = ax[0, 0]
    for (nn, s), d in sorted(runs.items(), key=lambda kv: -kv[0][1]):
        if nn != n: continue
        p = np.array([r["psi"] for r in d["rows"]])
        a.hist(np.clip(p, -0.3, 1.5), bins=120, histtype="step", lw=1.6,
               label=f"hfact $\\sigma$={s}  (operable {d['summary']['operable']:.3f})")
    a.axvline(0, color="k", lw=1); a.legend(fontsize=8)
    a.set_xlabel(r"$\psi$ = min$_z$ max$_j$ $g_j$"); a.set_title(r"feasibility function $\psi$")
    a = ax[0, 1]; a.hist(net[op], bins=80, color="tab:blue", alpha=.85); a.axvline(400, color="k", lw=1, ls="--")
    a.set_xlabel("MW"); a.set_title("net electric power | operable (400 MW required)")
    a = ax[0, 2]; a.hist(np.clip(coe[op], 0, 1500), bins=80, color="tab:green", alpha=.85)
    a.axvline(coe_p, color="k", lw=1, ls="--", label=f"PROCESS's {coe_p:.0f}"); a.legend(fontsize=8)
    a.set_xlabel("coe"); a.set_title("cost of electricity | operable")
    a = ax[1, 0]; a.hist(ph[op], bins=80, color="tab:purple", alpha=.85)
    a.axvline(75, color="k", lw=1, ls="--", label="file 75 MW"); a.axvline(inst, color="tab:red", lw=1, label=f"installed {inst:.0f} MW (heating + CD)")
    a.legend(fontsize=8); a.set_xlabel("MW"); a.set_title("heating power the balance asks for | operable")
    a = ax[1, 1]
    a.hist(psi[op], bins=60, alpha=.75, color="tab:blue", label="operable")
    a.hist(np.clip(psi[~op], -0.3, 1.5), bins=60, alpha=.75, color="tab:red", label="not")
    a.set_yscale("log"); a.set_xlabel(r"$\psi$"); a.legend(fontsize=8); a.set_title(r"$\psi$ split by outcome")
    a = ax[1, 2]
    from collections import Counter
    act = Counter(a_ for r, ok in zip(rows, op, strict=True) if ok for a_ in r["active"])
    act = {k: v / max(1, op.sum()) for k, v in act.items()}
    keys = sorted(act, key=lambda k: act[k])
    a.barh(keys, [act[k] for k in keys], color="tab:blue")
    a.set_xlabel("fraction of operable draws where active")
    a.set_title(f"active constraints, hfact $\\sigma$={sigma} (all movable by the operator)")
    plt.tight_layout(rect=(0, 0, 1, 0.96))
    path = f"{OUT}/{PREFIX}_histograms.png"
    plt.savefig(path, dpi=140)
    print("wrote", path)


def render_dsm():
    """The operator's problem as two interactive DSM pages (`flexibility_dsm.draw`):
    the `Optimise` over the three knobs inserted and `nested_inside` it, carrying only
    the conditions the operator can move -- cottax refuses a constant as a condition,
    which is how the 16 / 8 split is found rather than assumed."""
    import jax
    jax.config.update("jax_enable_x64", True)
    from cottax.pytree.executable import ExecutableGraph
    from cottax.pytree.names import PathMap
    from cottax.pytree.plan import Insert, Plan
    from cottax.pytree.problem import Optimise
    from flexibility_dsm import draw
    from common import OUT as OUT_DIR, process_reference
    from functional_process.cottax.architectures import closing, driven, ouu, session
    from functional_process.cottax.architectures.mda import assign_drivers, default_drivers
    from functional_process.cottax.queries import nested_inside
    from functional_process.configurations import kinds_tokamak as kinds

    live = session.open_session(NAME)
    proc = process_reference(NAME)
    dv, cv = deterministic_values(live, proc, driven.PAIRINGS)
    model = driven.two_stage(live, n=4, design_values=dv, closing_values=cv, installed=84.8,
                             freezes=driven.FREEZES)
    built = model.closed
    g = built.problem.graph  # undriven: `Assign` refuses a problem already driven
    knobs = tuple(v for v in built.design if v.spelling in kinds.KNOBS)
    varying = {n.spelling for n in model.stages.varying}
    owner = {v.spelling: n.spelling for v, n in g.graph.owners.items()}
    movable = tuple(c for c in model.constraints
                    if not isinstance(c, ouu.RecourseBound) and owner.get(c.spelling) in varying)
    print(f"optimised: {[v.spelling for v in knobs]}")
    print(f"movable by the operator ({len(movable)}):", [c.spelling.rsplit('.', 1)[-1] for c in movable])
    node = Optimise(objective=model.var_of[ouu.COE], unknowns=knobs, equalities=(),
                    inequalities=movable)
    graph = (Plan(g) + Insert(PathMap(((closing.OPTIMISE, node),)))).graph
    graph = nested_inside(graph, closing.OPTIMISE)
    drivers = default_drivers(graph)
    for place in dict.fromkeys(built.places.values()):
        drivers[place] = closing.safeguarded()
    drawn = ExecutableGraph(assign_drivers(graph, drivers))
    out = OUT_DIR / "dsm" / NAME
    out.mkdir(parents=True, exist_ok=True)
    files = draw(drawn, out, "flexibility_tokamak",
                 "the operator's problem: density, temperature and xenon optimised over a fixed "
                 "build and installed power; the heating power, the helium fraction and the "
                 "beta closing c2, c62 and c1 inside the MDA")
    print("wrote", files, f"({len(graph.nodes)} nodes) in {out}")


def main(argv=None):
    a = _parse(argv)
    if a.figures:
        return figures()
    if a.dsm:
        return render_dsm()
    cfg = {"n": a.n, "sigma": a.sigma, "jit_cache": a.jit_cache, "drop_c16": a.drop_c16,
           "no_freeze": a.no_freeze}
    t0 = time.perf_counter()
    state = _setup(cfg)
    np, model, places = state["np"], state["model"], state["places"]
    print(f"setup {time.perf_counter() - t0:.1f} s   hfact sigma {a.sigma}   knobs "
          f"{[o.rsplit('.', 1)[-1] for o in state['knobs']]}   installed {state['installed']:.1f} MW", flush=True)
    print(f"   movable conditions ({len(state['movable'])}): "
          f"{[state['names'][k] for k in state['movable']]}")
    print(f"   constant given the build ({len(state['fixed'])}): "
          f"{[state['names'][k] for k in state['fixed']]}", flush=True)
    x_fixed = np.asarray(model.x0, float)
    _compile(state, x_fixed)
    state["compiled"] = 0.0
    fixed = fixed_report(state, x_fixed)
    print("   build constraints at the design: " + ", ".join(f"{k} {v:+.2e}" for k, v in fixed.items()))
    if a.point:
        return point_report(state)
    point_lines = []
    point_report(state, out=lambda s: point_lines.append(s))
    print("\n".join(point_lines), flush=True)

    # The nominal world (row N, every belief at its nominal): the operator's own answer
    # there is the reference line, since PROCESS's reported point is not a cost optimum.
    z0 = np.ascontiguousarray(x_fixed[state["op"]])
    nominal = solve_sample(state, a.n, x_fixed, z0)
    print(f"   nominal world, operator re-optimised: operable {nominal['operable']}, psi {nominal['psi']:+.2e}, "
          f"coe {nominal['coe']:.2f} (PROCESS's reported {state['proc']['outputs']['.costs.coe']:.2f}), "
          f"net {nominal['net_mw']:.1f} MW, P_heat {nominal['p_heat']:.1f} MW, f_He {nominal['f_he']:.4f}, "
          f"n_e {nominal['ne']:.3g}, T_e {nominal['te']:.2f}, Xe {nominal['xe']:.2e}; active {nominal['active']}", flush=True)

    build = [i for i in range(len(places)) if i not in state["op"]]
    print("\n=== deterministic: " + ", ".join(
        f"{places[i].rsplit('.', 1)[-1]} {x_fixed[i]:.4g}" for i in build), flush=True)
    order = _order(model, a.n, a.order)
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
    summary = summarise(rows, state, a.n, coe_nominal=nominal["coe"])
    summary["fixed_conditions"] = fixed
    summary["point"] = point_lines
    summary["nominal"] = nominal
    ww = sorted(by_pid.values())
    print(f"   {wall:.1f} s total ({a.workers} workers, {len(res)} chunks); busy per worker "
          f"{min(ww):.1f}-{max(ww):.1f} s, compile {max(s['compile'] for s in res):.1f} s, "
          f"peak RSS/worker {max(s.get('rss_mb', 0) for s in res):.0f} MB", flush=True)
    tag = "" if not a.drop_c16 else "_noc16"
    tag += "" if not a.no_freeze else "_nofreeze"
    path = a.out or f"{OUT}/{PREFIX}_n{a.n}_s{a.sigma:.2f}_{a.order}{tag}.json"
    json.dump({"n": a.n, "places": places, "operating": list(state["knobs"]),
               "workers": a.workers, "order": a.order, "sigma": a.sigma,
               "movable": [state["names"][k] for k in state["movable"]],
               "fixed": [state["names"][k] for k in state["fixed"]],
               "design": x_fixed.tolist(), "wall": wall, "summary": summary, "rows": rows},
              open(path, "w"), indent=1)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
