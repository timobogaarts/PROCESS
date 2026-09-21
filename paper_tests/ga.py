"""A genetic algorithm over the machine, scored across sampled worlds.

**Where this sits.** `flexibility.py` asks, of PROCESS's one deterministic machine,
how it fares across the belief distribution. This asks the converse: is there a
*different* machine that fares better? It is the shape the literature calls
scenario-based (or "explicit averaging") robust optimisation -- Jin & Branke's 2005
survey, Sankary & Ostfeld 2018: every candidate is scored over a suite of sampled
scenarios, the suite is redrawn each generation, and the population's averaging is
what absorbs the resulting fitness noise. Nothing here is differentiated; the GA is a
black-box search whose one expensive primitive is the closed, lifted graph evaluated
across worlds -- either as **one batched MDA call** (the `vmap`ped program the OUU
study built, `architectures.ouu.two_stage` / `make`) or as **the operator's problem
solved per world** (`flexibility.solve_sample`, two-phase SLSQP), which is the
bilevel formulation: an outer population search over the build, an inner
optimisation over the operating point in every sampled world.

**Genome.** The build: B, R, the quench time, the copper fraction, and the
winding-pack width (exposed by `lift.lift_winding_pack`, so the coil is not re-sized
per sample). The operating point is the operator's, chosen per world, and how it is
chosen is `--recourse`:

- `solve` (the default): per world, `flexibility.solve_sample` -- phase 1 the
  feasibility function `psi = min_z max_j g_j`, phase 2 `min coe` over (T_e, the
  helium fraction) where operable. Exact, continuous, and the same evaluation the
  flexibility analysis reports for PROCESS's machine, so the two are comparable
  directly. ~0.06 s a draw per core, one compile per worker.
- `grid`: `ouu.two_stage(te_recourse=K)` -- K temperatures per world inside the one
  `vmap`, the cheapest feasible chosen per world, the helium fraction a shared gene.
  Coarser (K = 12 on 3..15 keV) and, at K = 12, no cheaper than `solve` on 12 cores.
- `none`: T_e and the helium fraction are genes shared across every world -- the
  no-recourse evaluation of the first prototype; it under-rates every design by
  what the operator could have recovered.

**Fitness.** `mean over N worlds of (charged cost if operable else PENALTY)`.
Operable is the flexibility analysis's definition: every driver converged, every
inequality within `G_TOL`, net > 0. No power floor. What a world is *charged*:

- `--charge rated` (the default): `coe x max(1, net / rated)` -- annual cost over
  `min(energy produced, rated energy)`. A world that under-produces pays its true
  (high) coe; one that over-produces is credited the rating and nothing more. This is
  PROCESS's own economics: in its problem `c16` is an *equality*, the plant is paid
  for its rating, and that equality is the only thing holding R at 26.7 m.
- `--charge actual`: coe as the cost model reports it, annual cost over the energy
  actually produced. It falls with plant size, so a search under it runs to the box
  bound on R -- and so does the *deterministic* optimiser once `c16` is relaxed to
  `net >= rated`: `n_equality=1` on the same problem converges to R = 30.0, quench
  time 50 (both bounds), B 5.38 T, coe 61.5 against 121.8 $/MWh (measured
  2026-09-20). "Winning by size" is therefore a property of the relaxation, not of
  uncertainty, and the honest baseline under this charge is that relaxed optimum:
  `--baseline relaxed` seeds generation 0 with it and certifies against it.

Three prototypes found the artefacts this design avoids: no floor and `actual`
(2026-09-19) gave a "2x cheaper" 2164 MW plant; a rated-power *floor* with `actual`
gave a 1564 MW plant at R = 30 "cheaper" at 105 against 116 $/MWh only because bigger;
the floor itself was a feasibility criterion the question never had.

**Cost.** `solve`: pop x N two-phase solves a generation over `--workers` processes
(~0.06 s a draw a core after one ~50 s compile per worker); 32 x 256 x 40 generations
is ~40 min on 12 cores. `grid`/`none`: one `run_batch` per candidate per generation.
`--certify` re-scores the winner and the baseline on one fresh, larger draw, which is
the number to quote -- a generation's best is selected on its own draw and biased low.

## Recipe

    export JAX_PLATFORMS=cpu PYTHONPATH=~/PROCESS:$SC/cottax_head/src:~/PROCESS/paper_tests
    $PY paper_tests/ga.py --pop 32 --n 256 --gens 40 --workers 12 --certify 2048
    $PY paper_tests/ga.py ... --charge actual --baseline relaxed      # size allowed
    $PY paper_tests/flexibility.py --n 4096 --workers 12 --order hfact \\
        --design-json paper_tests/out/ga_best.json                    # the winner, at N

Writes `out/ga_run.json` (every generation, the winner, the certification of both
designs on the same draw), `out/ga_best.json` (the winner in `--design-json` form) and
`out/ga.png`.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
NAME = "stellarator_helias"
HE = ".physics.f_nd_alpha_thermal_electron"


def _parse(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--pop", type=int, default=24)
    p.add_argument("--n", type=int, default=128, help="worlds per candidate per generation")
    p.add_argument("--gens", type=int, default=20)
    p.add_argument("--penalty", type=float, default=1000.0,
                   help="cost charged to a world the machine cannot be run in")
    p.add_argument("--sigma", type=float, default=0.10, help="hfact belief sigma")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--recourse", default="solve", choices=("solve", "grid", "none"),
                   help="how the operating point is chosen per world (module docstring)")
    p.add_argument("--te-grid", type=int, default=12, help="`grid`: T_e values per world")
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2),
                   help="`solve`: worker processes")
    p.add_argument("--charge", default="rated", choices=("rated", "actual"))
    p.add_argument("--c16", default="charge", choices=("charge", "closed", "none"),
                   help="what pins the plant class, since net power is an OUTPUT per "
                        "world and never a constraint. `charge`: `--charge rated`, a "
                        "world is credited at most the rated energy. `closed`: PROCESS's "
                        "own reading of c16 -- a design-time equality at the nominal "
                        "world, `net(nominal, operator's optimum) = rated`, carried as "
                        "the GA carries an equality: a tolerance band (`--rating-tol`) "
                        "and a graded penalty outside it; coe is then charged as reported. `none`: nothing; "
                        "size wins, and `--baseline relaxed` is the honest reference")
    p.add_argument("--rated", default="nominal",
                   help="the rated net electric MW: 'nominal' (the port's own at "
                        "PROCESS's design and set-points, 982.4), or a number")
    p.add_argument("--baseline", default="process", choices=("process", "relaxed"),
                   help="the design generation 0 is seeded with and the winner is "
                        "certified against: PROCESS's, or the deterministic optimum "
                        "with c16 relaxed to net >= rated")
    p.add_argument("--rating-tol", type=float, default=0.02,
                   help="`closed`: |net_nominal / rated - 1| allowed before the penalty")
    p.add_argument("--rating-hard", action="store_true",
                   help="`closed`: outside the band the candidate is dead (fitness = the "
                        "penalty) rather than penalised in proportion -- the equality as a "
                        "constraint, not a price. With the graded penalty the 2026-09-20 "
                        "run paid it and overbuilt to 17 %% over the rating anyway")
    p.add_argument("--certify", type=int, default=2048,
                   help="worlds of one fresh draw the winner and the baseline are both "
                        "re-scored on; 0 skips it")
    p.add_argument("--elite", type=int, default=2)
    p.add_argument("--spread", type=float, default=0.15,
                   help="lognormal sigma of generation 0 around the baseline design")
    p.add_argument("--mutation", type=float, default=0.06,
                   help="lognormal sigma of the per-gene mutation (the operable band in B "
                        "is ~2 % wide: under `closed` use ~0.02 and `--spread 0.05`)")
    p.add_argument("--jit-cache", default=os.path.join(HERE, "jitcache"))
    p.add_argument("--out", default=None)
    return p.parse_args(argv)


# ------------------------------------------------------------------ the baseline design
def baseline_values(live, which: str):
    """`(design_values, closing_values)` as `ouu.two_stage` takes them: PROCESS's own
    converged design (`common.deterministic_values`), or the port's MDF optimum of
    the same problem with `c16` moved from the equalities to the inequalities.
    """
    from common import deterministic_values
    if which == "process":
        return deterministic_values(live, "one")
    from functional_process import configurations
    from functional_process.configurations import kinds
    from functional_process.cottax.architectures import session
    from functional_process.cottax.architectures.sand import iteration_variable_path
    cfg = configurations.load(NAME)
    relaxed = dataclasses.replace(cfg, problem=dataclasses.replace(cfg.problem, n_equality=1))
    r = session.open_session(relaxed).solve("MDF")
    assert r["status"] == "converged", r["status"]
    x = {iteration_variable_path(i).spelling: float(v)
         for i, v in zip(cfg.problem.ixc, r["x"], strict=True)}
    closing = set(kinds.PAIRINGS["one"].values())
    print(f"relaxed deterministic optimum: objf {r['objf']:.4f} at "
          f"{ {k.rsplit('.', 1)[-1]: round(v, 4) for k, v in x.items()} }", flush=True)
    return [v for k, v in x.items() if k not in closing], {k: v for k, v in x.items() if k in closing}


def build_model(sigma, n, seed, te_recourse, baseline):
    import jax
    jax.config.update("jax_enable_x64", True)
    from functional_process.configurations import kinds
    from functional_process.cottax.architectures import lift, ouu, session
    live = session.open_session(NAME)
    dv, cv = baseline_values(live, baseline)
    beliefs = tuple(dataclasses.replace(b, a=sigma) if b.path == ".physics.hfact" else b
                    for b in kinds.BELIEFS)
    return ouu.two_stage(live, n=n, alpha=0.9, seed=seed, lifts=(lift.lift_winding_pack,),
                         beliefs=beliefs, te_recourse=te_recourse,
                         held=tuple(kinds.BUILD_LEAVES) + tuple(kinds.ECONOMIC),
                         design_values=dv, closing_values=cv), ouu


# ------------------------------------------------------------------ scoring
def charged(coe, net, ok, rated, charge, penalty):
    """What each world is charged; `rated` credits at most the rated energy."""
    cost = coe * np.maximum(1.0, net / rated) if charge == "rated" else coe
    return np.where(ok, np.clip(cost, 0.0, penalty), penalty), cost


def summary(coe, net, ok, cost, fit):
    return {"fitness": float(fit.mean()), "operable": float(ok.mean()),
            "cost_median": float(np.median(cost[ok])) if ok.any() else None,
            "coe_median": float(np.median(coe[ok])) if ok.any() else None,
            "net_median": float(np.median(net[ok])) if ok.any() else None,
            "cost": cost, "coe": coe, "net": net, "ok": ok}


class BatchScorer:
    """`grid` / `none`: one batched MDA call per design over the drawn worlds."""

    def __init__(self, model, ouu, a, rated):
        self.model, self.ouu, self.a, self.rated = model, ouu, a, rated
        fns = ouu.make(model, hoist=False)
        self.run, self.L = ouu.jitted(fns, "run_batch"), fns["layout"]
        self.places = [v.spelling for v in model.design]
        self.genes = np.arange(len(self.places))       # every design place is a gene

    def worlds(self, seed, n=None):
        if n is None:
            _, theta, starts = self.ouu.fresh_sample(self.model, seed)
            return theta, starts
        m = self.model
        _, theta = self.ouu.sample(m.beliefs, m.var_of, m.nominal, n, seed)
        return theta, np.tile(m.starts0[-1], (n + 1, 1))

    def score(self, x, worlds):
        import jax.numpy as jnp
        theta, starts = worlds
        y = np.asarray(self.run(jnp.asarray(x, float), theta, jnp.asarray(starts)))[:-1]
        L = self.L
        g0, g1 = L["c_g"]
        ok = (self.ouu.valid_rows(L, y) & (y[:, L["c_net"]] > 0)
              & np.all(y[:, g0:g1] <= self.ouu.G_TOL, axis=1))
        coe, net = y[:, L["c_coe"]], y[:, L["c_net"]]
        fit, cost = charged(coe, net, ok, self.rated, self.a.charge, self.a.penalty)
        return summary(coe, net, ok, cost, fit)


# -- `solve`: the operator's problem per world, in worker processes
_W = {}


def _worker_init(cfg, x_home):
    """Build the model and compile the per-draw programs once per process."""
    import flexibility
    state = flexibility._setup(cfg)
    flexibility._compile(state, np.asarray(x_home, float))
    _W.update(state=state, flexibility=flexibility, sample_seed=None, x_home=np.asarray(x_home, float))


def _worker_worlds(seed, n):
    """Redraw the worker's copy of the world set (the same Sobol' draw in every process)."""
    if _W["sample_seed"] == (seed, n):
        return
    state = _W["state"]
    model, ouu = state["model"], state["ouu"]
    Theta, theta = ouu.sample(model.beliefs, model.var_of, model.nominal, n, seed)
    starts = np.tile(model.starts0[-1], (n + 1, 1))
    state["model"] = dataclasses.replace(model, n=n, Theta=Theta, theta=theta, starts0=starts)
    j = [b.path for b in model.beliefs].index(".physics.hfact")
    _W["order"] = np.argsort(Theta[:n, j])          # walk in hfact order: warm-start locality
    _W["sample_seed"] = (seed, n)


def _nominal_net(x, z):
    """The operator's optimum in the nominal world (row `n` of the model's draw):
    net electric MW, or -1 where the machine cannot be run there."""
    state, fx = _W["state"], _W["flexibility"]
    r = fx.solve_sample(state, int(state["model"].n), np.asarray(x, float), z)
    return (r["net_mw"] if r["operable"] else -1.0), r


def _worker_nominal(args):
    """`--c16 closed`: the candidate's net electric MW at the operator's optimum in the
    nominal world, or -1 where it cannot be run there."""
    seed, n, x = args
    _worker_worlds(seed, n)
    z = np.ascontiguousarray(_W["x_home"][_W["state"]["op"]])
    return _nominal_net(np.asarray(x, float), z)[0]


def _rated_of(seed, n, x):
    """The baseline's own operator-optimal net at the nominal world."""
    _worker_worlds(seed, n)
    return _nominal_net(np.asarray(x, float), np.ascontiguousarray(_W["x_home"][_W["state"]["op"]]))[0]


def _worker_task(args):
    """One design over a slice of the world set: (index, operable, coe, net) rows."""
    seed, n, x, idx = args
    _worker_worlds(seed, n)
    state, fx = _W["state"], _W["flexibility"]
    x = np.asarray(x, float)
    z = np.ascontiguousarray(x[state["op"]])
    rows = []
    for i in _W["order"][idx]:
        r = fx.solve_sample(state, int(i), x, z)
        rows.append((int(i), r["operable"], r["coe"], r["net_mw"]))
        if r["operable"]:
            z = np.ascontiguousarray(np.array([r["te"], r["f_alpha"]], float))
    return rows


class SolveScorer:
    """`solve`: the two-phase operator's problem in every world, over a process pool."""

    def __init__(self, model, ouu, a, rated):
        import multiprocessing as mp
        from functional_process.configurations import kinds
        self.model, self.ouu, self.a, self.rated = model, ouu, a, rated
        self.places = [v.spelling for v in model.design]
        self.op = [self.places.index(kinds.TE), self.places.index(HE)]
        self.genes = np.array([i for i in range(len(self.places)) if i not in self.op])
        cfg = {"n": a.n, "sigma": a.sigma, "knobs": "both", "jit_cache": a.jit_cache,
               "require_net": 0.0, "restarts": 0, "baseline": a.baseline}
        self.x_home = np.asarray(model.x0, float)
        self.pool = mp.get_context("spawn").Pool(
            a.workers, initializer=_worker_init, initargs=(cfg, self.x_home))
        self.chunks = 2
        if a.c16 == "closed":
            # the plant class: what the BASELINE delivers at the nominal under the same
            # operator -- PROCESS's design freed of the equality makes ~1165 MW there,
            # not the 982.4 it is held to
            seed0 = a.seed + 999
            self.rated = float(self.pool.apply(_rated_of, (seed0, a.n, self.x_home)))
            print(f"c16 closed at the nominal: the baseline's operator-optimal net there is "
                  f"{self.rated:.1f} MW; a candidate is a plant of that class when its own "
                  f"is within {100 * a.rating_tol:.0f} %, and penalised in proportion "
                  f"outside", flush=True)

    def worlds(self, seed, n=None):
        return (seed, n or self.a.n)

    def nominal(self, xs, seed, n):
        """`--c16 closed`: each candidate's operator-optimal net in the nominal world."""
        return self.pool.map(_worker_nominal, [(seed, n, x) for x in xs], chunksize=1)

    def score_many(self, xs, worlds):
        seed, n = worlds
        xs = [self._full(x) for x in xs]
        nominal = self.nominal(xs, seed, n) if self.a.c16 == "closed" else None
        skip = [nominal is not None and nominal[j] < 0 for j in range(len(xs))]
        tasks, owner = [], []
        for j, x in enumerate(xs):
            if skip[j]:
                continue
            for idx in np.array_split(np.arange(n), self.chunks):
                tasks.append((seed, n, x, idx)); owner.append(j)
        results = self.pool.map(_worker_task, tasks, chunksize=1)
        out = []
        for j in range(len(xs)):
            if skip[j]:                    # not a plant at all: inoperable at the nominal
                nan = np.full(n, np.nan)
                out.append(summary(nan, nan, np.zeros(n, bool), nan,
                                   np.full(n, self.a.penalty)))
            else:
                rows = sorted(r for t, o in zip(results, owner, strict=True) if o == j for r in t)
                ok = np.array([r[1] for r in rows]); coe = np.array([r[2] for r in rows])
                net = np.array([r[3] for r in rows])
                fit, cost = charged(coe, net, ok, self.rated, self.a.charge, self.a.penalty)
                out.append(summary(coe, net, ok, cost, fit))
            out[-1]["x"] = xs[j].tolist()
            if nominal is not None:
                # the rating as the GA carries an equality: free inside the band,
                # the penalty scaled by the excess outside it
                off = abs(nominal[j] / self.rated - 1.0) if nominal[j] > 0 else 1.0
                out[-1]["net_nominal"] = float(nominal[j])
                if self.a.rating_hard and off > self.a.rating_tol:
                    out[-1]["fitness"] = self.a.penalty
                else:
                    out[-1]["fitness"] += self.a.penalty * max(0.0, off - self.a.rating_tol)
        return out

    def _full(self, x):
        full = self.x_home.copy()
        full[self.genes] = np.asarray(x, float)[self.genes]
        return full

    def score(self, x, worlds):
        return self.score_many([x], worlds)[0]

    def close(self):
        self.pool.close(); self.pool.join()


# ------------------------------------------------------------------ the GA
def evolve(a, scorer, rng):
    """Elitism, tournament selection, BLX-alpha crossover, lognormal mutation; the
    world suite redrawn every generation. Returns the history and the final population.
    """
    model = scorer.model
    genes = scorer.genes
    lo, hi = np.asarray(model.lower, float), np.asarray(model.upper, float)
    x_base = np.asarray(model.x0, float)
    k = len(x_base)
    pop = np.clip(x_base * rng.lognormal(0.0, a.spread, size=(a.pop, k)), lo, hi)
    fixed = np.setdiff1d(np.arange(k), genes)
    pop[:, fixed] = x_base[fixed]                  # non-genes stay at the baseline
    pop[0] = x_base                                # the baseline itself in generation 0
    history, began = [], time.perf_counter()
    for gen in range(a.gens):
        worlds = scorer.worlds(a.seed + 1000 * (gen + 1))
        if hasattr(scorer, "score_many"):
            scored = scorer.score_many(pop, worlds)
        else:
            scored = [scorer.score(ind, worlds) for ind in pop]
        fit = np.array([s["fitness"] for s in scored])
        order = np.argsort(fit)
        b = order[0]
        history.append({"gen": gen, "best_fitness": float(fit[b]),
                        "best_operable": scored[b]["operable"],
                        "best_cost_median": scored[b]["cost_median"],
                        "best_coe_median": scored[b]["coe_median"],
                        "best_net_median": scored[b]["net_median"],
                        "mean_fitness": float(fit.mean()),
                        "best_genome": scored[b].get("x", pop[b].tolist()),
                        "best_net_nominal": scored[b].get("net_nominal"),
                        "baseline_fitness": float(fit[0]) if gen == 0 else None})
        print(f"gen {gen:3d}  best {fit[b]:8.2f}  (operable {scored[b]['operable']:.3f}, "
              f"charged {scored[b]['cost_median'] or float('nan'):7.2f}, "
              f"coe {scored[b]['coe_median'] or float('nan'):7.2f}, "
              f"net {scored[b]['net_median'] or float('nan'):7.1f}"
              + (f", nominal {scored[b]['net_nominal']:7.1f})" if "net_nominal" in scored[b] else ")")
              + "   "
              f"mean {fit.mean():8.2f}   {time.perf_counter() - began:6.1f} s", flush=True)
        nxt = [pop[i].copy() for i in order[:a.elite]]
        while len(nxt) < a.pop:
            p, q = (pop[min(rng.choice(a.pop, 3), key=lambda i: fit[i])] for _ in range(2))
            w = rng.uniform(-0.25, 1.25, size=k)                     # BLX-alpha
            child = np.where(rng.random(k) < 0.9, w * p + (1 - w) * q, p)
            child = np.clip(child * rng.lognormal(0.0, a.mutation, size=k), lo, hi)
            child[fixed] = x_base[fixed]
            nxt.append(child)
        pop = np.array(nxt)
    return history, pop, scorer.places, x_base


def certify(a, scorer, x_ga, x_base, places):
    """Both designs on one fresh draw of `--certify` worlds -- the same worlds."""
    worlds = scorer.worlds(a.seed + 777, a.certify)
    out = {}
    for label, x in ((a.baseline, x_base), ("ga", x_ga)):
        s = scorer.score(x, worlds)
        c = s["cost"][s["ok"]]
        out[label] = {"fitness": s["fitness"], "operable": s["operable"],
                      "cost_median": s["cost_median"], "coe_median": s["coe_median"],
                      "net_median": s["net_median"], "net_nominal": s.get("net_nominal"),
                      "cost_p10_p90": [float(np.percentile(c, 10)), float(np.percentile(c, 90))]
                                      if s["ok"].any() else None,
                      "x": {p.rsplit(".", 1)[-1]: float(v) for p, v in zip(places, x, strict=True)},
                      "_cost": c}
        print(f"certified {label:8s}: fitness {s['fitness']:8.2f}  operable "
              f"{s['operable']:.3f}  charged median {s['cost_median']}  coe median "
              f"{s['coe_median']}  net median {s['net_median']}", flush=True)
    return out


def figure(history, cert, path, a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    gens = [h["gen"] for h in history]
    a1.plot(gens, [h["best_fitness"] for h in history], "-o", ms=3, label="generation best")
    a1.plot(gens, [h["mean_fitness"] for h in history], "-", alpha=0.6, label="population mean")
    a1.axhline(history[0]["baseline_fitness"], ls="--", c="k", lw=1,
               label=f"{a.baseline} design (gen 0 draw)")
    a1.set_xlabel("generation"); a1.set_ylabel("fitness  (mean charged cost)"); a1.legend()
    if cert and all(len(v["_cost"]) for v in cert.values()):
        allc = np.concatenate([v["_cost"] for v in cert.values()])
        bins = np.linspace(np.percentile(allc, 0.5), np.percentile(allc, 99.5), 60)
        for (label, r), c in zip(cert.items(), ("C0", "C1"), strict=True):
            a2.hist(r["_cost"], bins=bins, alpha=0.5, color=c,
                    label=f"{label}: operable {r['operable']:.2f}, median {r['cost_median']:.1f}")
        a2.set_xlabel(f"charged cost ({a.charge}) in the operable worlds, same draw, "
                      f"recourse {a.recourse}")
        a2.set_ylabel("worlds"); a2.legend()
    fig.tight_layout(); fig.savefig(path, dpi=140); print("wrote", path)


def main(argv=None) -> int:
    a = _parse(argv)
    if a.c16 != "charge" and a.charge == "rated":
        a.charge = "actual"                    # the rating is in the design, not the charge
    model, ouu = build_model(a.sigma, a.n, a.seed, a.te_grid if a.recourse == "grid" else 0,
                             a.baseline)
    # the rating: the port's own net power at PROCESS's design and set-points, in the
    # nominal world (982.4 MW), from a program without any recourse
    if a.rated == "nominal":
        m0, _ = build_model(a.sigma, a.n, a.seed, 0, "process")
        fns0 = ouu.make(m0, hoist=False)
        rated = float(np.asarray(fns0["run_batch"](np.asarray(m0.x0, float), m0.theta,
                                                    m0.starts0))[-1, fns0["layout"]["c_net"]])
        del m0, fns0
    else:
        rated = float(a.rated)
    print(f"rated {rated:.1f} MW; charging {a.charge}; recourse {a.recourse}; "
          f"baseline {a.baseline}", flush=True)
    scorer = (SolveScorer if a.recourse == "solve" else BatchScorer)(model, ouu, a, rated)
    print(f"genes: {[scorer.places[i].rsplit('.', 1)[-1] for i in scorer.genes]}", flush=True)
    rng = np.random.default_rng(a.seed)
    history, pop, places, x_base = evolve(a, scorer, rng)
    x_ga = np.asarray(history[-1]["best_genome"], float)
    cert = certify(a, scorer, x_ga, x_base, places) if a.certify else None
    if hasattr(scorer, "close"):
        scorer.close()
    short = [p.rsplit(".", 1)[-1] for p in places]
    print(f"\n{a.baseline:8s}:", dict(zip(short, np.round(x_base, 4).tolist(), strict=True)))
    print("GA      :", dict(zip(short, np.round(x_ga, 4).tolist(), strict=True)))
    os.makedirs(OUT, exist_ok=True)
    path = a.out or os.path.join(OUT, "ga_run.json")
    json.dump({"pop": a.pop, "n": a.n, "gens": a.gens, "penalty": a.penalty,
               "sigma": a.sigma, "seed": a.seed, "rated_mw": rated, "charge": a.charge,
               "recourse": a.recourse, "te_grid": a.te_grid, "baseline": a.baseline, "c16": a.c16,
               "places": places, "baseline_design": x_base.tolist(), "ga_design": x_ga.tolist(),
               "history": history,
               "certified": {k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")}
                             for k, v in cert.items()} if cert else None},
              open(path, "w"), indent=1)
    json.dump({"label": "ga_best", "places": places, "x": x_ga.tolist()},
              open(os.path.join(OUT, "ga_best.json"), "w"), indent=1)
    figure(history, cert, os.path.splitext(path)[0] + ".png", a)
    print("wrote", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
