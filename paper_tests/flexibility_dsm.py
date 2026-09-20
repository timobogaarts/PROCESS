"""The graph the flexibility analysis solves, and the recipe that builds it.

Every step is cottax ops on the previous graph. Counts are measured
(`--dsm` renders steps 0-6; the appendix is `scratchpad/bsplit/`).

| # | step | ops | nodes | inputs | problems |
|---|---|---|---|---|---|
| 0 | the declared machine | `indat.graph_for(machine)` | 154 | 305 | 4 |
| 1 | minus unbacked nodes | `Delete(.vacuum.duct_diameter_root_find)` | 152 | 300 | 3 |
| 2 | the MDA | `FixedPointCut` x2 + `Assign(drivers)` | 154 | 300 | 5 |
| 3 | + constraints, objective | `Insert(icc x14)` + `Insert(Optimise)` | 170 | 305 | 6 |
| 4 | + `c2` closed by the density | `Insert(RootFind)` + `Nest` | 170 | 312 | 6 |
| 5 | + the winding pack lifted | `Undrive`+`Unnest`+`Undetermine`+`Delete`+`Insert` | 171 | 313 | 5 |
| 6 | + the operator's problem | `Insert(Optimise)` + `nested_inside` | 172 | | |

Step 2 applies only **2** of `mda.CUTS`' nine: the other seven cycles do not exist in
this stellarator's graph. The problems it leaves are the four the models declare
themselves (`proton_rate_density.cycle`, `f_ster_div_single`,
`profiles.ion_vol_avg_temperature`, `power.delta_eta_step`) plus
`stellarator.coils.intersect` -- the winding-pack sizing rule, which step 5 takes apart.

Step 4 adds no node but seven boundary inputs: the closing mints the root find's ports.
Step 5 *removes* a problem (the coil stops solving for its own width) and adds one node
(the inserted signed residual, `^cond.lift.stellarator.wp_width_r_min`).

Step 6 is this file. The `Optimise` owns exactly two unknowns -- `T_e` and the helium
fraction -- and carries only the **six** conditions the operator can move. The other
seven (c82, c83, c32, c34, c35, c65 and the lifted pack rule) are constants once the
build is fixed, and cottax refuses them outright: *"reads ... as a condition, but
nothing in its cycle produces it, so no unknown of the problem can move it -- a constant
is not a condition"*. That refusal is how the six/seven split was found rather than
assumed, and it matches the measured spans (c67 11.2, c8 7.7, c18 3.6, c62 2.7, c24 2.4,
c17 1.5; the other seven exactly 0.0000).

## Appendix: the field split, measured and rejected

`b_plasma_toroidal_on_axis` reaches 122 of 154 nodes -- the plasma *and* the whole magnet
chain -- so lowering it re-sizes the magnet. The magnet-sizing cone is 9 nodes and the
field enters it through exactly one reader, so separating the built field from the
operated one is two ops:

    Insert(.Split.field)   an identity owning B, read from a new input  (B is a boundary
                           input, and `Cut` only splits *computed* places)
    Cut(B, readers=(stellarator_scaling_factors,))   mints the copy the magnet reads

Deterministically inert: `objf 1.3351987988619283` either way, and the lifted pack width
seeds at 0.7169 either way. `stages.leaves` then refuses the graph until each new place
is given a kind, which is the modelling claim made explicit. Plumbing it needs beyond the
two ops: re-point ixc 2, re-key `reference.bounds`, pin the built field into the
configuration.

**Result: worth nothing here.** With `B_operating <= B_design` enforced the operator picks
the maximum field in every world and operability is unchanged (0.4102 -> 0.4102). Without
that cap it appears to gain 47 %, which is a 4.70 T magnet being run at 6.08 T.
"""
import jax
jax.config.update("jax_enable_x64", True)

import dsms
from common import OUT, deterministic_values
from cottax.answerable import AnswerableGraph
from cottax.names import PathMap
from cottax.plan import Insert, Plan
from cottax.problem import Optimise
from functional_process.cottax.architectures import closing, lift, ouu, session
from functional_process.cottax.architectures.mda import assign_drivers, default_drivers
from functional_process.cottax.queries import nested_inside
from functional_process.configurations import kinds

OPERATING = (kinds.TE, ".physics.f_nd_alpha_thermal_electron")

def render():
    live = session.open_session("stellarator_helias")
    dv, cv = deterministic_values(live, "one")
    model = ouu.two_stage(live, n=4, alpha=0.9, seed=0, lifts=(lift.lift_winding_pack,),
                          held=tuple(kinds.BUILD_LEAVES) + tuple(kinds.ECONOMIC),
                          design_values=dv, closing_values=cv)
    built = model.closed
    operating = tuple(v for v in built.design if v.spelling in OPERATING)
    fixed = tuple(v for v in built.design if v.spelling not in OPERATING)
    assert len(operating) == 2, [v.spelling for v in built.design]
    print("optimised:", [v.spelling for v in operating])
    # `built.design` is the closed problem's, so it still carries `hfact` (ixc 10),
    # which is a *belief* sampled per draw rather than a build number. The five real
    # build places are the rest.
    BELIEF = ".physics.hfact"
    print("held fixed (build):", [v.spelling for v in fixed if v.spelling != BELIEF], flush=True)
    print("sampled per draw   :", [v.spelling for v in fixed if v.spelling == BELIEF], flush=True)

    # Only the conditions the operator can actually move belong to the operator's
    # problem. With the build fixed the other seven are constants, and cottax refuses a
    # constant as a condition -- which is how this split was found rather than assumed.
    g = built.problem.graph
    owner = {v.spelling: n for n, d in g.definitions.items() for v in getattr(d, "owns", ())}
    readers = [n for n, d in g.definitions.items()
               if any(v.spelling in OPERATING for v in getattr(d, "reads", ()))]
    reach = {n.spelling for n in g.graph.descendants(tuple(readers))} | {n.spelling for n in readers}
    movable = tuple(c for c in built.report["inequalities"]
                    if owner.get(c.spelling) is not None and owner[c.spelling].spelling in reach)
    fixed_conds = [c.spelling for c in built.report["inequalities"] if c not in movable]
    print(f"movable by the operator ({len(movable)}):", [c.spelling.rsplit('.',1)[-1] for c in movable])
    print(f"constant given the build ({len(fixed_conds)}):", [c.rsplit('.',1)[-1] for c in fixed_conds], flush=True)

    node = Optimise(
        objective=built.report["objective"],
        unknowns=operating,
        equalities=(),
        inequalities=movable,
    )
    graph = (Plan(built.problem.graph) + Insert(PathMap(((closing.OPTIMISE, node),)))).graph
    graph = nested_inside(graph, closing.OPTIMISE)
    drivers = default_drivers(graph)
    for cond, place in built.places.items():
        drivers[place] = closing.with_bounds(closing.bracketed(), built.pairings[cond],
                                             built.session.reference.bounds)
    drawn = AnswerableGraph(assign_drivers(graph, drivers))

    out = OUT / "dsm" / "stellarator_helias"
    out.mkdir(parents=True, exist_ok=True)
    files = dsms.draw(drawn, out, "flexibility",
                      "the operator's problem: T_e and the helium fraction optimised over a "
                      "fixed build, the density closing the power balance inside the MDA")
    print("wrote", files, f"({len(graph.nodes)} nodes)")


if __name__ == "__main__":
    render()
