---
kind: model-unit
status: draft
confidence: high
---

**Ported (4/4).** `coils.py` / `test_coils.py`: `j_crit_cable_from_fraction` and
`bmax_from_awp` (tier-1, ported previously), `intersect` (tier-2, ported an earlier
pass), and `jcrit_from_material` (tier-1 per branch, split into 8 pure functions + 8
`ExplicitFunction` nodes — see below). The blocker recorded in earlier snapshots of this
file (`process/models/superconductors.py` not being an audited registry unit) is
resolved: that module is now ported as registry unit #22
(`functional_process/models/physics/superconductors.py`/`.md`), which is what unblocked
that pass — read that record's own "cottax node" sketch first, it is the design that
pass finalized almost unchanged.

**This pass: `intersect` gains a genuine `ImplicitFunction`/`RootFind` declaration,
`Intersect`, plus a concrete test-only `AbstractDriver`, `IntersectBisectionNewtonPolish`
(`coils.py`).** The plain `intersect(x1, y1, x2, y2, xin)` function is unchanged and kept
— `Intersect` is a *second*, structural way to reach the same answer, not a replacement.
See "cottax node" below for the full design and `_audit/next_steps.md` §7 for why this is
worth doing even though that section's own earlier conclusion ("nothing else needs
`intersect`'s internal unknowns, so leave it eager") stands unchanged: the reason here is
different — making the root-find's *solver* a first-class, swappable `Drive` choice
instead of something hardcoded inside `intersect`'s own body.

## source
`process/models/stellarator/coils/coils.py` (303 lines, full file in scope). 4 module-
level functions: `j_crit_cable_from_fraction`, `jcrit_from_material`, `intersect`,
`bmax_from_awp`. Called only from `coils/calculate.py` (registry unit #9, `st_coil`'s
winding-pack solve loop) — no calls in the other direction, and no calls between these
four functions and `forces.py`/`mass.py`/`quench.py` (grepped: none of those three files
import from `coils.py` or vice versa; each of the four files in this batch is called
directly by unit #9, not by each other).

## data footprint

| VarPath | read/write | classification | note |
|---|---|---|---|
| — | — | — | `j_crit_cable_from_fraction` takes no `data` at all — already pure in the source. |
| `.stellarator_config.stella_config_a1` | read | explicit-arg (via `data` back-door) | `bmax_from_awp` |
| `.stellarator_config.stella_config_a2` | read | explicit-arg (via `data` back-door) | same |
| — | — | — | `intersect` takes no `data` at all — already pure in the source (see below). |

`jcrit_from_material` itself is also already `data`-free in the source (its 11 arguments
are all plain scalars — confirmed by reading the whole function body, no `self.data`
access anywhere) — the "footprint" below is instead the real call-site provenance of
those 11 arguments, from `winding_pack_total_size` (`process/models/stellarator/coils/
calculate.py:404-430`, unit #9's real call site), and (for two branches) the port's own
minted `VarPath`s where PROCESS has none:

| VarPath | read/write | classification | note |
|---|---|---|---|
| `.tfcoil.j_crit_sc` | write (minted) | — | the return value of `jcrit_from_material` (`j_crit_sc * 1e-6`) has no PROCESS storage location at all — it's consumed immediately, still inside the same sampling loop, by `winding_pack_total_size`'s `lhs = f_j_tf_wp_critical_max * jcrit_vector`. Minted per `coilcurrent`'s precedent (`calculate.md`). |
| `.tfcoil.b_max` | read (minted) | local-intermediate | one element (`b_max_k[k]`) of `bmax_from_awp`'s own 200-point sample vector — a solve-loop local, not an established field, same status `coils.md`'s earlier snapshot gave `bmax_from_awp`'s own call-site arguments. |
| `.tfcoil.t_helium` | read (minted) | local-intermediate | `data.tfcoil.tftmp + data.tfcoil.tmargmin`, computed once per call, constant across the sampling loop — a derived local, not itself a field. |
| `.tfcoil.j_tf_wp` | read | explicit-arg | real, established field — already an `Input` on `calculate.py`'s `WindingPackTotalSize`/`WindingPackJTfWp` under this name. Only read by branch 2 (Bi-2212). |
| `.tfcoil.f_a_tf_turn_cable_space_extra_void` | read | explicit-arg | real field, read by branch 2. |
| `.tfcoil.fhts` | read | explicit-arg | real field (source parameter name `f_hts`), read by branch 2. |
| `.tfcoil.f_a_tf_turn_cable_copper` | read | explicit-arg | real field, read by branch 2. |
| `.tfcoil.bcritsc` | read | explicit-arg | real field (source parameter name `b_crit_sc`), read by branch 4 only. |
| `.tfcoil.tcritsc` | read | explicit-arg | real field (source parameter name `t_crit_sc`), read by branch 4 only. |
| `.tfcoil.b_crit_upper_nbti` | read | explicit-arg | real field, read by branch 7 only. |
| `.tfcoil.t_crit_nbti` | read | explicit-arg | real field, read by branch 7 only. |

Branches 1, 3, 5, 8 read only the two minted locals (`b_max`, `t_helium`) plus fixed
literals (`bc20m`/`tc0m`/`c0`, hard-coded per branch, not `data` reads at all); branch 6
reads only the two minted locals. This confirms the earlier snapshot's reads-set claim
directly rather than by inference.

## proposed signature(s)

Ported, tier-1, as written (see `coils.py`):
```python
def j_crit_cable_from_fraction(j_crit_sc, f_tf_conductor_copper, f_he) -> float:
    ...

def bmax_from_awp(
    wp_width_radial, current, n_tf_coils, r_coil_major, r_coil_minor,
    stella_config_a1, stella_config_a2,
) -> float:
    ...
```

Ported, tier-2, this pass:
```python
def intersect_residual(x, x1, y1, x2, y2) -> float:
    """y1_interp(x) - y2_interp(x); vanishes at the crossing."""
    ...

def intersect(x1, y1, x2, y2, xin) -> float:
    """The x at which the two curves cross, found by bisection + Newton polish."""
    ...
```
`intersect` was already pure in the source — no `data.*` reads, no calls to any other
model. Its args are exactly PROCESS's: `x1, y1, x2, y2` (tabulated curves), `xin`
(starting guess).

**Ported this pass — `jcrit_from_material`, split one function per `i_tf_sc_mat`
branch**, per `traceability_policy.md`'s split-by-default and the reads-set evidence
below (also independently confirmed by `superconductors.md`'s own audit of the material
models this dispatch calls). Each branch calls its already-ported material model
(`functional_process.models.physics.superconductors`) directly with that branch's real
argument set; the common `* 1e-6` scaling applied once at the very end of the source's
`if`/`elif` chain is folded into each function's own `return` instead of factored out
separately (nothing left over to share once the material-model call itself differs
per branch):

```python
def jcrit_from_material_iter_nb3sn(t_helium, b_max) -> float: ...              # i_tf_sc_mat == 1
def jcrit_from_material_bi2212(t_helium, b_max, j_tf_wp,
    f_a_tf_turn_cable_space_extra_void, fhts, f_a_tf_turn_cable_copper) -> float: ...  # == 2
def jcrit_from_material_nbti_lubell(t_helium, b_max) -> float: ...             # == 3
def jcrit_from_material_iter_nb3sn_user_defined(
    t_helium, b_max, bcritsc, tcritsc) -> float: ...                           # == 4
def jcrit_from_material_wst_nb3sn(t_helium, b_max) -> float: ...               # == 5
def jcrit_from_material_rebco(t_helium, b_max) -> float: ...                   # == 6
def jcrit_from_material_gl_nbti(
    t_helium, b_max, b_crit_upper_nbti, t_crit_nbti) -> float: ...             # == 7
def jcrit_from_material_gl_rebco(t_helium, b_max) -> float: ...                # == 8
```

Each returns `j_crit_sc` (MA/m2), matching the source's `j_crit_sc * 1e-6`. Verified
directly against the real `jcrit_from_material` dispatcher (branch by branch, at
hand-picked points spanning both sides of every `b_max > bc20m` guard) before writing
the harness cases — see `test_coils.py`. Branch 6 (REBCO) is the one exception: the
source's own call site (`process/models/stellarator/coils/coils.py:136`) calls
`superconductors.jcrit_rebco(t_helium, b_max, 0)` with **three** positional arguments to
a function that takes exactly **two** — executing this branch as written raises
`TypeError` unconditionally (confirmed directly; also flagged independently by
`superconductors.md`'s open question 1). `jcrit_from_material_rebco` calls
`jcrit_rebco`'s real 2-argument signature instead — not a faithful reproduction of a call
that cannot execute, since there is nothing faithful to reproduce — matching the
treatment `calculate.py`'s `_critical_current_density_by_material` (a local, unaudited
restatement of this same dispatch, written to unblock `winding_pack_total_size` before
this unit's own port existed) already gave this branch. **`calculate.py`'s
`_critical_current_density_by_material` is a known, deliberate duplicate of this
function set, investigated and kept as two independent implementations** — a later
consolidation pass checked whether `winding_pack_curves`'s 200-point sampling could call
these functions directly instead, and found it would either regress the REBCO branch
(this file reproduces PROCESS's real 3-argument-call bug there; `calculate.py`'s local
copy deliberately does not, so the port has *a* working REBCO branch) or stop faithfully
reproducing that bug -- so the duplication stays, documented, not merged. See
`calculate.md`'s own note on this and `total_process.py`'s module docstring.

## cottax node

**Actually written** for `j_crit_cable_from_fraction`/`bmax_from_awp`'s siblings in other
units, but **not** for either of those two functions themselves — for the same reason in
both cases. Every one of their real call-site arguments (`coilcurrent`, `wp_width_r_min`,
`r_coil_major`, `r_coil_minor` for `bmax_from_awp`; `j_crit_sc`/`f_tf_conductor_copper`/
`f_he` for `j_crit_cable_from_fraction`, called from inside `jcrit_from_material`) is a
*local* computed inside `winding_pack_total_size`'s solve loop (`coils/calculate.py`,
unit #9), not an established `.area.field` this audit has independently verified —
wrapping either as a node now would assert a wiring this pass has no basis for (see
`schema.md`: "skip this section... while open questions about the signature itself are
unresolved"). Correct home for both is wherever unit #9 declares its own solve —
`calculate.md`'s open question #1 already raises this exact tension for `coilcurrent`.

**`intersect` gains a real `ImplicitFunction`/`RootFind` declaration, `Intersect`, this
pass** — the sketch this section used to carry (below, superseded) is now real code,
almost unchanged from the draft:

```python
class Intersect(ImplicitFunction):
    wp_width_r_min = Output(lambda s: s.stellarator.wp_width_r_min)

    def residual(
        self,
        wp_width_r_min=Input(lambda s: s.stellarator.wp_width_r_min),
        wp_width_r=Input(lambda s: s.stellarator.wp_width_r),
        lhs=Input(lambda s: s.stellarator.lhs),
        rhs=Input(lambda s: s.stellarator.rhs),
    ):
        return intersect_residual(wp_width_r_min, wp_width_r, lhs, wp_width_r, rhs)
```

Minted `VarPath`s, unchanged from the draft's own choice: `.stellarator.wp_width_r`
(used as both `x1` and `x2`, matching the real call site's `intersect(wp_width_r, lhs,
wp_width_r, rhs, ...)`), `.stellarator.lhs`, `.stellarator.rhs` (the two sampled
curves), and `.stellarator.wp_width_r_min` (the crossing point — this node's one
declared `Output`/unknown). None of these four have a PROCESS storage location; all are
minted per `coilcurrent`'s precedent (`calculate.py`'s `CoilCurrent`). `to_graph(
Intersect())` builds cleanly into exactly two nodes (the `residual` body, a
`CallableNode`, and the `RootFind` problem it feeds) — confirmed directly, not just
asserted (`test_coils.py::test_intersect_declares_a_body_and_a_root_find_problem`).

**`.stellarator.wp_width_r_min` is minted but not ungrounded** (added by the constraint-32
investigation, `_audit/constraint_32_investigation.md`). PROCESS stores no
`wp_width_r_min` field, but it *does* store the same number one line later: after the
turn-size clamp (`process/models/stellarator/coils/calculate.py:465`), `awp_rad =
wp_width_r_min` is written straight into `data.tfcoil.dr_tf_wp_with_insulation`
(`calculate.py:481,489`). So `.tfcoil.dr_tf_wp_with_insulation` *is* this unknown's
converged value whenever the clamp is inactive — measured on `stellarator_helias`,
`dx_tf_turn_general**2 = 3.136e-03` against `0.7170`, inactive by 228× — and a lower
bound otherwise. `mda_harness.KNOWN_MINT_VALUES` now uses exactly that as the `RootFind`'s
**starting guess**, which is what let `Intersect`'s whole 4-node SCC come out of
`mda_harness.EXCLUDED_NODE_NAMES`. The port's solved answer is then compared back against
the same field like any other output, and agrees to the harness's `rtol=1e-6` — the first
PROCESS-comparable number this node has ever had (its `Tier2Contract` deliberately has
none, see below).

`residual` reads the unknown itself back (`wp_width_r_min`, the same `VarPath` its own
`Output` declares), the same shape as `~/jaxgraph`'s own
`test_interfaces_pytree_namespace.py::Disc1` example — **not** a self-loop:
`residual`'s `CallableNode` only *reads* the real `.stellarator.wp_width_r_min` (the
current guess) and *writes* `^cond.stellarator.wp_width_r_min`; a separate, bodyless
`RootFind` problem node (minted at `^problem.Intersect`) is what actually owns the real
`.stellarator.wp_width_r_min` — confirmed directly
(`test_coils.py::test_intersect_body_reads_the_unknown_back_without_owning_it`).

**Why now, given `_audit/next_steps.md` §7 already concluded `intersect` needs no
follow-up**: §7's own test ("does anything else in the graph need to read or write
something inside this iteration's own state") is unchanged and still says no — nothing
else needs `intersect`'s internal unknowns visible. The reason to declare `Intersect`
anyway is a different one: as a plain function, `intersect`'s algorithm (bisection
bracket, then Newton polish) is baked into the leaf itself, with no way for the graph to
say *how* that block is solved, let alone swap the answer. As an `ImplicitFunction`/
`RootFind` pair, the algorithm becomes a `Drive`'s `driver` argument — a first-class,
inspectable, replaceable choice, structurally separate from *what* must vanish
(`residual`) — without the *declaration* itself committing to any one choice: undriven,
`Intersect` is a perfectly valid, if unproducing, `RootFind` problem sitting in the
graph (`Graph.declared`), same status as any other undriven Shape A/B problem node
`_audit/next_steps.md` §5 already tracks.

**`IntersectBisectionNewtonPolish`, a concrete `AbstractDriver` (`coils.py`), for
tests only.** Wraps `intersect`'s own bisection-then-Newton-polish algorithm exactly:
pulls the two curve arrays out of `conditions.context` (the block's own closed-over
external inputs — `Intersect`'s only unknown is `wp_width_r_min`, so `wp_width_r`/`lhs`/
`rhs` are exactly its `context`) and calls `intersect` directly, seeding `xin` from
`start[0]` when a guess is given, otherwise the curves' median. This is what lets a test
build a real `Drive` (`schedule_for(to_graph(Intersect()), {Intersect().problem_name:
IntersectBisectionNewtonPolish()})`) and get a real converged number, while `Intersect`
itself commits to no particular algorithm — the driver is one legitimate answer among
others, not registered anywhere as *the* answer. Verified directly, not just
constructed: driving with a sample's own `xin` as the starting guess reproduces
`intersect`'s own answer exactly, over every curated sample in `_intersect_samples()`
(`test_coils.py::test_intersect_bisection_newton_polish_drives_to_the_same_answer_as_intersect`).
Some `_crossing_curve_case` samples have more than one genuine crossing in-domain
(only the sign of the curve endpoints is pinned, not the interior), so bisection's own
answer can genuinely depend on where it starts — the test seeds the same `xin` the
sample's own reference call used, rather than claiming any starting guess reaches the
same root.

**`xin` still has no port on `Intersect`, confirmed directly, not merely asserted**
(open question 2, below, now resolved by direct construction rather than left as a
prediction): `residual` declares exactly 4 `Input`s (the unknown plus the two curves),
one fewer than `intersect`'s own 5-argument signature
(`test_coils.py::test_intersect_has_no_port_for_xin`) — a `RootFind`'s starting guess
comes from whatever `Drive`s the block (`evaluate.py`'s `Drive.__call__`: `guess =
env[unknowns] if started else None`), never from an `In` on `residual` itself. `intersect`
the plain function still takes and uses `xin` faithfully; only the *node* wrap has one
fewer declared input than the function it wraps.

**Not wired into `coils/calculate.py`'s `winding_pack_total_size` by this pass beyond
what `calculate.md` itself does** — `Intersect` is designed so that unit #9's own
`winding_pack_total_size` split (`WindingPackIntersectInputs`/`WindingPackTotalSizePost`,
`calculate.py`) can reuse this exact class unmodified as the middle piece of a three-node
block, rather than needing its own wrapper: `WindingPackIntersectInputs` mints
`.stellarator.wp_width_r`/`.lhs`/`.rhs` at exactly the `VarPath`s `Intersect` reads. See
`calculate.md`'s own "cottax node" section for that composition.

Superseded sketch, kept only as a historical record of what this section drafted before
it was real code:

```python
# class Intersect(ImplicitFunction):
#     wp_width_r_min = Output(lambda s: s.stellarator.wp_width_r_min)
#
#     def residual(
#         self,
#         x1=Input(lambda s: s.stellarator.wp_width_r),
#         y1=Input(lambda s: s.stellarator.lhs),
#         x2=Input(lambda s: s.stellarator.wp_width_r),
#         y2=Input(lambda s: s.stellarator.rhs),
#     ):
#         return intersect_residual(self.owns[0], x1, y1, x2, y2)  # sketch only
```
The draft's `self.owns[0]` for the unknown was never valid (`NodalDeclaration` has no
`owns` property; `DeclaredNode.owns` belongs to the *problem*, not the body) — the real
`residual` reads the unknown back as an ordinary `Input` on the same `VarPath` its
`Output` declares instead, per `Disc1`'s own precedent, confirmed above.

**`jcrit_from_material`'s 8 branches are the exception — 8 `ExplicitFunction` nodes
written this pass**, one per `i_tf_sc_mat` value (`JcritIterNb3sn`, `JcritBi2212`,
`JcritNbtiLubell`, `JcritIterNb3snUserDefined`, `JcritWstNb3sn`, `JcritRebco`,
`JcritGlNbti`, `JcritGlRebco` — `coils.py`). Unlike the three functions above, this was
written despite the same "real arguments are solve-loop locals" situation applying to
two of its own inputs (`b_max`, `t_helium` — see the data-footprint table), because the
*other* six of its eleven source arguments genuinely are established `.tfcoil.*` fields
(`j_tf_wp`, `f_a_tf_turn_cable_space_extra_void`, `fhts`, `f_a_tf_turn_cable_copper`,
`bcritsc`/`tcritsc`, `b_crit_upper_nbti`/`t_crit_nbti`) — this dispatch is not uniformly
"nothing but locals" the way the other three functions are, and the task that produced
this pass specifically asked for the real port, not a further deferral. `b_max`/
`t_helium` and the shared output `.tfcoil.j_crit_sc` are therefore **minted**, following
`coilcurrent`'s precedent (`calculate.py`'s `CoilCurrent`) rather than blocked on: mint
what has no established field, wire what does.

All 8 classes mint the identical output `VarPath` (`.tfcoil.j_crit_sc`) — exactly what
`configuration.py`'s `Switch.check_arms_are_exclusive` needs to accept them as one
mutually-exclusive `Switch` group, syntactically. **Updated, later consolidation pass:
investigated and deliberately still not registered — not an oversight, a real structural
gap.** `.tfcoil.t_helium`/`.tfcoil.b_max` (this sketch's own `Input`s, and the real
classes') are locals of `winding_pack_curves`'s 200-point sampling loop
(`b_max = b_max_k[k]`, an *array* value per sample, not a single scalar `VarPath`), and
PROCESS has exactly one real call site for the whole `jcrit_from_material` dispatch,
inside that same sampling loop (confirmed: the only call in
`process/models/stellarator/coils/calculate.py` is its own
`jcrit_vector[k] = jcrit_from_material(...)`, per-sample). There is no single-point
scalar evaluation of this dispatch anywhere PROCESS calls it, so registering the `Switch`
sketched below would assert a wiring that does not exist, not fill in a real one — left
unregistered on purpose. If a future port genuinely needs one, minting a *reporting-only*
scalar evaluation (e.g. at the resolved crossing point, post-`Intersect`) would be a new
design decision, not a mechanical consolidation step. Still purely mechanical if that
decision is ever made:

```python
Switch(
    path=".tfcoil.i_tf_sc_mat",
    default=1,  # `tfcoil_variables.py:246`
    alternatives=(
        Alternative(value=1, declarations=(JcritIterNb3sn,)),
        Alternative(value=2, declarations=(JcritBi2212,)),
        Alternative(value=3, declarations=(JcritNbtiLubell,)),
        Alternative(value=4, declarations=(JcritIterNb3snUserDefined,)),
        Alternative(value=5, declarations=(JcritWstNb3sn,)),
        Alternative(value=6, declarations=(JcritRebco,)),
        Alternative(value=7, declarations=(JcritGlNbti,)),
        Alternative(value=8, declarations=(JcritGlRebco,)),
    ),
)
```
— one `Alternative` per class above, no further code, once `total_process.py`'s own
consolidation pass is ready to touch it (not done here, per this pass's explicit
boundary — see "open questions").

**Updated — this is no longer a sketch.** `coils.py` now has a real `Intersect`
(`ImplicitFunction`) class, registered in `total_process.py` and forming a genuine
4-node SCC with `calculate.py`'s `WindingPackIntersectInputs`/`WindingPackTotalSizePost`
(see `calculate.md`'s own "cottax node" section, and `total_process.py`'s registration
comment). The sketch below is kept for the historical record of the shape it converged
to; the real class's own docstring is the current source of truth.

The natural declaration, once unit #9 mints real `VarPath`s for its locals (the same
minting `calculate.py`'s `CoilCurrent` node already did for `coilcurrent`), is a
pytree-namespace `ImplicitFunction` pairing `intersect_residual` with a `RootFind` over
one unknown:
```python
class Intersect(ImplicitFunction):
    wp_width_r_min = Output(lambda s: s.stellarator.wp_width_r_min)

    def residual(
        self,
        x1=Input(lambda s: s.stellarator.wp_width_r),
        y1=Input(lambda s: s.stellarator.lhs),
        x2=Input(lambda s: s.stellarator.wp_width_r),
        y2=Input(lambda s: s.stellarator.rhs),
    ):
        return intersect_residual(<the unknown>, x1, y1, x2, y2)  # sketch only
```
**Open question this sketch surfaces**: `xin` has no place in this shape at all. A
`RootFind`'s starting guess comes from whatever `Drive`s the block (see `evaluate.py`'s
`Drive.__call__`: `guess = env[unknowns] if started else None`, handed to the driver
positionally), not from an `In` on the residual body — so PROCESS's `xin` argument simply
does not survive into the node-graph port as a port. That is fine for `intersect` itself
(the pure function below still takes and uses it, faithfully), but it means the *node*
wrap, once written, has one fewer declared input than the function it wraps — worth
flagging explicitly, since `_audit/naming_convention.md` has no category yet for "an
argument that is real in the pure function but has no port in the node."

## tier signal
- `j_crit_cable_from_fraction`: **tier 1** — pure, no `data`, no branch.
- `bmax_from_awp`: **tier 1** — pure once the 2-field `data` back-door is closed (same
  shape as `st_sudo_density_limit` in `density_limits.py`).
- `jcrit_from_material`: **tier 1, per branch — ported this pass.** Every one of the 8
  split functions is an explicit pure function: no internal iteration, no `self.data`
  access (the source's own 11-argument signature is already `data`-free), each calling
  exactly one already-ported (registry unit #22) material model plus, for branches 1/3,
  a `jnp.where`-guarded continuous branch and an unconditional floor (see "JAX-difficulty
  flags"). Confirms `superconductors.md`'s own tier characterization of the functions
  this dispatch calls, and matches the split-worthy contrast case `next_steps.md` § 1
  already names this switch as ("8 genuinely different reads-sets, no shared body to
  speak of — the *opposite* shape [from a formula-only switch kept static]").
- `intersect`: **tier 2, ported an earlier pass.** Self-contained (no calls into other
  models, no `data` access at all — confirmed by reading the full 100-odd lines): a
  genuine internal Newton-Raphson-style solve over two tabulated `(x, y)` curves, fixed
  100-iteration cap with an early `break` on `abs(y01 - y02) < epsy`. Exactly the "internal
  iterative loop closing over state local to one model" `test_harness.md`'s tier-2 section
  describes, and — per that same section's own framing — this is the first unit in the
  registry to actually exercise `Tier2Contract`'s residual-based pass criterion (see
  `test_coils.py`).

## switches touched
- `i_tf_sc_mat` (`.tfcoil.i_tf_sc_mat`, `tfcoil_variables.py:246`, default 1) — **a
  topology-changing switch, resolved this pass, not yet in `switches.md`'s original 10.**
  Per `naming_convention.md`'s "switches are not ports": read once to decide which of the
  8 `ExplicitFunction` classes exists in the assembled graph, not carried as a `VarPath`
  or a static kwarg on one composite node — the mechanism is `total_process.py`'s
  `Switch`/`Alternative` (`configuration.py`), same as `i_plasma_pedestal`/`i_bldgs_size`.
  **Split decision: split, confidence high** — full reads-set now confirmed directly (not
  inferred) by reading both this file's `jcrit_from_material` and
  `superconductors.md`'s own independent trace of the same dispatcher (both agree): branch
  4 alone reads `bcritsc`/`tcritsc`, branch 7 alone reads `b_crit_upper_nbti`/
  `t_crit_nbti`, branches 1/3/5/8 use fixed literals, branches 2/6 use neither — see the
  data-footprint table above. **Entangled switches**: none found. Also read (same field)
  inside `mass.py`'s `superconductor()` as a plain array index
  (`data.tfcoil.dcond[i_tf_sc_mat - 1]`) — a data-table lookup, not a formula branch; see
  `mass.md`'s note on the same field for why that use is treated differently.
- `intersect` touches no switch — genuinely pure over its five array/scalar arguments.

## calls into other models
- `jcrit_from_material`'s 8 ported functions each call exactly one of
  `functional_process.models.physics.superconductors.{itersc, bi2212, jcrit_nbti,
  western_superconducting_nb3sn, jcrit_rebco, gl_nbti, gl_rebco}` (registry unit #22,
  ported and audited — see `superconductors.md`); `itersc`/`western_superconducting_nb3sn`
  transitively call that unit's own `bottura_scaling` helper, not a second-level call this
  file's functions make directly.
- Neither `intersect` nor the two previously-ported functions call anything outside this
  file.

## JAX-difficulty flags
- `jcrit_from_material_iter_nb3sn`/`jcrit_from_material_nbti_lubell` (branches 1, 3),
  **resolved this pass**: the source's `if b_max > bc20m: j_crit_sc = 1.0e-9 else:
  itersc(...)`/`jcrit_nbti(...)` is a data-dependent branch on a *continuous* traced
  value, not a switch — `minor`, `needs-lax-cond-or-where`. Ported as `jnp.where`, always
  evaluating the material-model call on both branches (verified `jax.jacfwd`-finite at the
  boundary via `--fp-gradients`, not just value-checked) then selecting, followed by the
  source's own unconditional `max(1.0e-9, j_crit_sc)` floor.
- `jcrit_from_material_rebco` (branch 6): the source's real call site passes an extra
  positional argument `jcrit_rebco` does not accept (`TypeError`, unconditional — see
  the "proposed signature(s)" section above and `superconductors.md`'s open question 1)
  — not a JAX-traceability issue at all, a pre-existing PROCESS bug in unreachable code.
  Ported by calling the correct 2-argument signature; `minor`/informational, not a
  blocker, but worth flagging loudly since it means this one branch has no PROCESS
  reference answer to check the port's *value* against (`test_coils.py`'s reference
  wrapper for this branch calls `superconductors.jcrit_rebco` directly instead of going
  through the broken dispatcher — see `_reference_jcrit_rebco`'s docstring there).
- No other JAX-difficulty found across the remaining 6 branches (2, 4, 5, 7, 8) — each is
  a straight-line call into one already-ported, already-`jnp.where`-guarded material
  model (`superconductors.md`'s own JAX-difficulty flags cover the guards *inside* those
  functions; nothing extra is introduced by `jcrit_from_material`'s own dispatch code
  around them).
- `intersect`, **resolved an earlier pass**: the source's `for _i in range(100): ... break`
  loop with a data-dependent early exit had no faithful `jax`-traceable translation (JAX
  has no early-exit `break`), and its post-loop clamp-and-`logger.error` bail-out on
  leaving `[xmin, xmax]` is a diagnostic side effect on a data-dependent condition — not
  traceable as written either. The port does not attempt a line-for-line translation of
  either: it re-poses the same defining equation (`intersect_residual`) and drives it with
  `optimistix.root_find`'s `Bisection` (bracket = the curves' full x-overlap, which is a
  valid sign-changing bracket exactly whenever a crossing exists there — no `xin`-
  dependent windowing, so, unlike PROCESS's local Newton-Raphson, a bad `xin` cannot walk
  the solve off the domain or onto the wrong crossing) followed by a fixed few `jax.grad`-
  based Newton corrections once bisection has localised `x` into the correct linear
  segment of the piecewise-linear interpolation (see `coils.py`'s
  `_intersect_newton_polish` docstring for why a Newton step is *exact*, not approximate,
  once there). `throw=False` on the `root_find` call means non-convergence is reported
  through `optimistix`'s own `result` field rather than raised or logged — the traced
  equivalent of PROCESS's `logger.error`, per `traceability_policy.md`.
- `jnp.interp` (used by `intersect_residual`) is piecewise-linear and therefore not
  everywhere-differentiable (a kink at each tabulated `x` node) — `minor`, and not a
  concern for this port specifically: `Tier2Contract` has no gradient-agreement test (see
  `test_harness.md`), so no check here differentiates through the interpolation's kinks.
  Worth flagging for whoever eventually drives the `ImplicitFunction`/`RootFind` pair
  sketched above through an actual `jacfwd`, e.g. for a tier-4 MDA's own sensitivities.

## open questions
1. **[RESOLVED this pass]** ~~Should `process/models/superconductors.py` be added as a
   new registry unit?~~ Done, as unit #22 (`functional_process/models/physics/
   superconductors.py`/`.md`), which is what unblocked `jcrit_from_material`'s own port
   this pass.
2. **[RESOLVED, a later pass]** ~~`xin`'s disappearance from the node wrap~~ — `Intersect`
   (`## cottax node` above) is now real code, and the disappearance is confirmed
   directly rather than left as a naming-convention gap: `residual` declares 4 `Input`s
   against `intersect`'s 5-argument signature
   (`test_coils.py::test_intersect_has_no_port_for_xin`). Still worth a real
   `naming_convention.md` category for "an argument that is real in the pure function
   but has no port in the node that wraps it" — not added here (out of this file's own
   boundary), just confirmed as a real, reproducible instance rather than a prediction.
3. **`i_tf_sc_mat`'s 8 branches — whether all are actually reachable from real
   stellarator input files**: still not checked (would need a `preset_config.py`/
   input-file survey, out of this pass's scope too) — the split into 8 nodes this pass
   makes this question sharper, not different: an unreachable branch would now be an
   unreachable `Alternative` rather than an unreachable `elif`, same underlying fact.
4. **`calculate.py`'s `_critical_current_density_by_material` is now a known duplicate**
   of this file's 8 functions — both implement the identical `i_tf_sc_mat` dispatch,
   calling the identical material models, with identical branch-6 bug workaround. Not
   rewired here (out of this unit's boundary — `calculate.py` is read-only for this
   pass, and several `FixedPointFunction` conversions landed there this same session that
   a rewiring pass shouldn't risk disturbing incidentally). **Flagged for whoever does the
   consolidation pass**: once `total_process.py` wires up the `Switch`/`Alternative` group
   sketched in `## cottax node` above, `winding_pack_total_size`
   (`WindingPackIntersectInputs`/`WindingPackTotalSizePost`/`WindingPackJTfWp` in
   `calculate.py`) should plausibly be rewired to read `.tfcoil.j_crit_sc` from the real
   nodes here instead of calling its own
   local `_critical_current_density_by_material` — but that rewiring changes
   `calculate.py`, is a design decision about how a per-`k`-sample scalar node interacts
   with `winding_pack_total_size`'s 200-point vectorised sampling loop (this file's nodes
   are declared for one `(b_max, t_helium)` point, matching `jcrit_from_material`'s own
   real signature; `calculate.py` currently vectorises the dispatch with `jax.vmap`-free
   plain array ops inside `winding_pack_curves`), and is explicitly not done here.
5. **`total_process.py` registration**: per this task's explicit instruction, the 8 nodes
   above are written and tested but **not** assembled into a `Switch` or added to
   `total_process.py` — see `## cottax node` for the exact `Switch`/`Alternative` shape a
   consolidation pass should use; it is mechanical from what's written here.
