---
kind: model-unit
status: reviewed
confidence: high
---

**Ported (3 units).** `vacuum.py` / `test_vacuum.py`: `calculate_vacuum_pumping_simple`
(tier-1), `solve_duct_diameter`/`duct_diameter_residual` (tier-2, isolated Newton
root-find), `_solve_vacuum_pumping_old`/`calculate_vacuum_pumping_old` (tier-2, the full
`"old"` duct-sizing model). All three have passing `Tier1Contract`/`Tier2Contract`
harness tests (`functional_process/models/test_vacuum.py`). `VacuumVessel` (the file's
second class) is **out of scope** -- confirmed unreached from `Stellarator.run()`, see
below.

**Also added: a genuine `ImplicitFunction` for `solve_duct_diameter`.**
`DuctDiameterRootFind` (`vacuum.py`) declares the same defining equation
(`duct_diameter_residual`) as a real, undriven `RootFind` block -- two minted nodes
(the residual body and the `RootFind` problem), built and driven for test purposes only
(`test_vacuum.py`'s `test_duct_diameter_root_find_*`). See "cottax node: `DuctDiameterRootFind`" below for the full account, including why this
supersedes `next_steps.md` §7's earlier "no follow-up needed" finding for this one unit
specifically (by explicit instruction, not by re-deriving that finding), and the
corrected framing for `solve_duct_geometry`'s outer shrink loop (kept eager, see
"the outer shrink loop" below).

## source
`process/models/vacuum.py`, 995 lines, entry point `Vacuum.run()` (registry unit #16).
Traced in full:

- `Vacuum.run()` (32-96) -- dispatches on `.vacuum.i_vacuum_pumping` (`"old"`/
  `"simple"`, a topology-changing switch) to one of two essentially disjoint
  computations.
- `Vacuum.vacuum_simple()` (98-147) + `Vacuum._vacuum_simple_output()` (149-195) --
  the `"simple"` branch. Reporting shell excluded from scope.
- `Vacuum.vacuum()` (197-458), `Vacuum._newton_method_duct_diameter()` (460-501),
  `Vacuum._newton_function()` (503-526, `@staticmethod`), `Vacuum._write_to_outfile()`
  (528-734) -- the `"old"` branch. Reporting shell excluded from scope.
- `VacuumVessel` (736-995) -- **out of scope**, see below.

Two module-level imports worth recording since they're read but not computed here:
`process.models.build.FwBlktVVShape` and
`process.models.engineering.ivc_functions.{dshellvol, eshellvol}`, both used only by
`VacuumVessel`.

### `VacuumVessel` is unreached on the stellarator pipeline

`process/main.py:668-669` (`Models.__init__`) constructs both `self.vacuum = Vacuum()`
and `self.vacuum_vessel = VacuumVessel()`, and line 940 calls
`self.vacuum_vessel.output()` from `main.py`'s own (tokamak/general) output path.
`process/models/stellarator/stellarator.py`'s `Stellarator.__init__` (lines 63-99) is
injected `vacuum: Vacuum` but has **no** `vacuum_vessel` parameter at all -- grepped, no
`self.vacuum_vessel` or bare `VacuumVessel` reference anywhere under
`process/models/stellarator/`. `Stellarator.st_fwbs` computes its own internal
vacuum-vessel volume inline instead (`stellarator.py:1299`, "Internal vacuum vessel
volume" -- registry's chunk-1 synthesis record calls this "S5
`cryostat_and_vv_geometry`", already flagged as `self-contained, portable now` under
unit #1, not this one). `VacuumVessel` is therefore provably dead code on the
stellarator pipeline, the same shape as `avail_st`'s suspected (but here *confirmed*)
unreachability -- out of this unit's scope entirely, not audited further.

## data footprint

### `Vacuum.run()` (dispatcher)

| VarPath | read/write | classification | note |
|---|---|---|---|
| `.vacuum.i_vacuum_pumping` | read | topology-changing switch | `"old"`/`"simple"`, resolved at graph-assembly time, not a port -- see switches below |
| `.physics.molflow_plasma_fuelling_required` | read | explicit-arg | feeds `gasld` (local) |
| `.physics.m_fuel_amu` | read | explicit-arg | feeds `gasld` |
| `.build.dr_fw_plasma_gap_inboard`, `.build.dr_fw_plasma_gap_outboard` | read | explicit-arg (composite) | `dsol = 0.5*(a+b)`, local to `run()`, sole use is `vacuum()`'s `dsol` argument |
| `.build.r_shld_inboard_inner`, `.build.dr_shld_vv_gap_inboard`, `.build.dr_vv_inboard` | read | explicit-arg (composite) | `ritf = r_shld_inboard_inner - dr_shld_vv_gap_inboard - dr_vv_inboard`, local, sole use is `vacuum()`'s `ritf` argument |
| `.physics.p_fusion_total_mw`, `.physics.rmajor`, `.physics.rminor`, `.physics.a_plasma_surface`, `.physics.vol_plasma`, `.build.dr_shld_outboard`, `.build.dr_shld_inboard`, `.build.dr_tf_inboard`, `.tfcoil.n_tf_coils`, `.times.t_plant_pulse_dwell`, `.physics.nd_plasma_electrons_vol_avg`, `.divertor.n_divertors` | read | explicit-arg | passed straight through to `vacuum()` |
| `.vacuum.n_vv_vacuum_ducts`, `.vacuum.dlscal`, `.vacuum.m_vv_vacuum_duct_shield`, `.vacuum.dia_vv_vacuum_ducts` | write | explicit-arg | `vacuum()`'s tuple return, unpacked directly onto `data` |
| `.vacuum.n_vac_pumps_high` | write | local-intermediate | `math.floor(pumpn + 0.5)`, `pumpn` being `vacuum()`'s first return value -- straight-line, no branch, same call |
| `.vacuum.n_iter_vacuum_pumps` | write | explicit-arg | `vacuum_simple()`'s return value, `"simple"` branch only |

### `Vacuum.vacuum_simple()`

| VarPath | read/write | classification | note |
|---|---|---|---|
| `.physics.molflow_plasma_fuelling_required` | read | explicit-arg | |
| `.vacuum.molflow_vac_pumps` | read | explicit-arg | |
| `.vacuum.volflow_vac_pumps_max` | read | explicit-arg | |
| `.vacuum.f_a_vac_pump_port_plasma_surface` | read | explicit-arg | |
| `.vacuum.f_volflow_vac_pumps_impedance` | read | explicit-arg | |
| `.physics.a_plasma_surface` | read | explicit-arg | |
| `.tfcoil.n_tf_coils` | read | explicit-arg | |
| `.vacuum.outgasfactor` | read | explicit-arg | |
| `.vacuum.pres_vv_chamber_base` | read | explicit-arg | |
| `.vacuum.outgasindex` | read | explicit-arg | |
| `.times.t_plant_pulse_dwell` | read | explicit-arg | |

No writes inside `vacuum_simple()` itself -- its return value is what `run()` writes
(see above). `n_iter_vacuum_pumps`/`npumpdown` (the two candidates combined by `max`)
are locals with no `VarPath`, used only for display in `_vacuum_simple_output`
(out of scope) -- dropped from the port's return, same convention as
`divertor.py`'s reporting-only intermediates.

### `Vacuum.vacuum()` (the `"old"` model)

All explicit-args (the function's own parameter list, per its docstring) plus these
`self.data.*` reads inside the body:

| VarPath | read/write | classification | note |
|---|---|---|---|
| `.vacuum.i_vacuum_pump_type` | read | formula-changing switch | picks `sp` (pump speed table) and the final `pumpn *= 2` cryopump multiplier; same reads-set both values -- see switches below |
| `.vacuum.i_vac_pump_dwell` | read | formula-changing switch, data-entangled | picks `tpump`'s formula, but the branch condition also depends on the *runtime value* of `t_plant_pulse_dwell` (`i_vac_pump_dwell==1 or t_plant_pulse_dwell==0`) -- not resolvable as a pure compile-time switch, see switches below |
| `.vacuum.pres_vv_chamber_base` | read | explicit-arg | |
| `.vacuum.pres_div_chamber_burn` | read | explicit-arg | |
| `.vacuum.outgrat_fw` | read | explicit-arg | |
| `.vacuum.temp_vv_chamber_gas_burn_end` | read | **dead read** | feeds `pend`, which only ever appears as `pend/pstart` with `pstart = 0.01*pend` -- see "Real finding" below. Does not affect any of the five outputs. |
| `.times.t_plant_pulse_coil_precharge` | read | explicit-arg | |
| `.physics.p_fusion_total_mw`, `.physics.temp_plasma_electron_vol_avg_kev` | read | diagnostic-only | read inside `_newton_method_duct_diameter` *only* to build a `logger.error` message on the (data-dependent) Newton non-convergence path -- no effect on any returned value. Dropped from the pure signature. |

Writes: the function's own 5-tuple return
`(pumpn, n_vv_vacuum_ducts, dlscal, m_vv_vacuum_duct_shield, dia_vv_vacuum_ducts)`.
`nplasma`/`ndiv`/`qtorus`/`gasld` etc. are ordinary parameters, not `self.data` reads
(the caller, `Vacuum.run()`, resolves those -- see its own table above).

No `implicit-io-via-callee` or `redundant-duplicate-write` anywhere in this file.

## Real findings (documented, not fixed -- per project policy, cf. `radiation_power.md`)

1. **`pend`/`pstart`'s ratio is a disguised constant.** `pend = 0.5*nplasma*k*
   temp_vv_chamber_gas_burn_end`; `pstart = 0.01*pend`. The only further use is
   `math.log(pend/pstart)`, which is `math.log(100)` **identically**, for any
   `pend != 0` -- `nplasma` and `.vacuum.temp_vv_chamber_gas_burn_end` cannot change
   any of `vacuum()`'s five returned values through this path (only the *displayed*
   `pend`/`pstart` numbers in `_write_to_outfile`, out of port scope). Confirmed
   algebraically, not just empirically. The pure port drops both from its signature and
   uses `jnp.log(100.0)` directly (see `_solve_vacuum_pumping_old`'s docstring).
2. **`pumpn2` is provably dominated by `pumpn1`.** `ccc = 2*s[i]/nduct`, so
   `nduct/s[i] - 1/ccc = nduct/s[i] - nduct/(2*s[i]) = nduct/(2*s[i])`, giving
   `pumpn1 = 2*s[i]/(sp[i]*nduct)` algebraically -- and `pumpn2 = 1.01*s[i]/(sp[i]*
   nduct)`. Since `2 > 1.01` and every factor is positive, `pumpn1 > pumpn2`
   unconditionally, so `max(pumpn, pumpn1, pumpn2) == max(pumpn, pumpn1)` always;
   `pumpn2` never controls the result. Not pruned from the port (kept for fidelity to
   the source, and the computation is free) -- flagged here in case the ratio `1.01`
   was meant to be compared against something other than `pumpn1`.
3. **`.vacuum.m_vv_vacuum_duct_shield` (`mvdsh`) is provably always `0.0`.**
   `thdsh = 0.0` is hardcoded ("Set duct shield thickness to zero for no biological
   shielding instead of thshldo/3.0e0", source comment at `vacuum.py:284-286`), and
   `arsh = 0.25*pi*((d*1.2 + thdsh)**2 - (d*1.2)**2)` is then `(x+0)**2 - x**2 = 0`
   identically for any `d`. Confirmed both algebraically and empirically (300-point
   fuzz sweep and the `test_old_model` legacy point both give `mvdsh == 0.0` exactly).
   Not a bug -- the field is a live, still-written `VarPath`, just structurally inert
   as the source currently stands; worth knowing before spending effort minting a node
   port for it that could never disagree with `0.0`.
4. **PROCESS's own duct-diameter Newton loop stops before its true root**, verified on
   the `test_old_model` legacy point (helium species, `i=2`): PROCESS reports
   `dimax = 0.42414752916950604`. Evaluating PROCESS's *own* `_newton_function` at that
   exact diameter gives a conductance of `3.778057`, not the target `ceff = 3.771849`
   it was solving for -- a genuine `~0.16%` residual, not machine round-off. One more
   Newton step (still using PROCESS's own formula) lands at `0.4238565`, matching this
   port's converged answer (`0.4238904`) to `~0.008%`. PROCESS's `dd <= 0.01` stopping
   test is a coarse 1%-relative-step cutoff, not an accuracy target -- it exits with a
   diameter whose own defining equation is still off by an order of magnitude more than
   the *next* step would have closed. Not a bug (`0.01` is presumably an intentional
   engineering tolerance, and the ETR-era model does not claim tighter accuracy), but
   it is the reason this port's `solve_duct_diameter` defaults to a much tighter
   `tol=1e-10` rather than reproducing `0.01` -- see that function's docstring and
   `test_vacuum.py`'s `_reference_solve_duct_diameter`/`TestSolveDuctDiameter`, which
   both exercise and document this exact discrepancy.
5. **Name collision**: `Vacuum.vacuum_simple`'s local `n_iter_vacuum_pumps` (the
   ITER-cryopump-throughput estimate, one of two candidates combined by `max`) and the
   `data` field `.vacuum.n_iter_vacuum_pumps` (which ends up holding the *combined*
   `npump`, not this local) share a name but not a value. Purely a naming footgun, not
   a computation bug -- `Vacuum.run()`'s assignment (`vp.n_iter_vacuum_pumps =
   self.vacuum_simple(output=output)`) is correct, just easy to misread.
6. **`ceff[imax]` and `d[imax]` can go briefly inconsistent** in the "space limited"
   (`nflag = 1`) exit of `_newton_method_duct_diameter`'s outer loop: `d[i]` is computed
   from the pre-shrink `ceff[i]`, then `ceff[i] *= 0.9` happens *after*, and if that
   shrunk value trips the `<= 1.1*s[i]` giveup check, the loop exits with `d[i]`
   corresponding to the old `ceff[i]`, not the one now stored. A real (if narrow) quirk
   of the source; not reproduced identically -- see `solve_duct_geometry`'s docstring
   for how the port's `ceff_used`/`ceff_final` split keeps a residual check meaningful
   either way, and the "open questions" section below for why this port's fuzz samples
   don't specifically target the regime where it would show up.

## proposed signature(s)

```python
def calculate_vacuum_pumping_simple(
    molflow_plasma_fuelling_required,
    molflow_vac_pumps,
    volflow_vac_pumps_max,
    f_a_vac_pump_port_plasma_surface,
    f_volflow_vac_pumps_impedance,
    a_plasma_surface,
    n_tf_coils,
    outgasfactor,
    pres_vv_chamber_base,
    outgasindex,
    t_plant_pulse_dwell,
) -> float:  # npump, -> .vacuum.n_iter_vacuum_pumps
    ...


def solve_duct_diameter(
    l1, l2, l3, xmult_i, ceff_i, max_iter=100, tol=1e-10
) -> (
    float
): ...  # tier-2, see "cottax node" -- not directly a VarPath producer, see below


def calculate_vacuum_pumping_old(
    p_fusion_total_mw,
    rmajor,
    rminor,
    dsol,
    a_plasma_surface,
    vol_plasma,
    dr_shld_outboard,
    dr_shld_inboard,
    dr_tf_inboard,
    ritf,
    n_tf_coils,
    t_plant_pulse_dwell,
    n_divertors,
    qtorus,
    gasld,
    i_vac_pump_dwell,
    i_vacuum_pump_type,
    pres_vv_chamber_base,
    pres_div_chamber_burn,
    outgrat_fw,
    t_plant_pulse_coil_precharge,
) -> tuple[float, float, float, float, float]:
    # (n_vac_pumps_high, n_vv_vacuum_ducts, dlscal, m_vv_vacuum_duct_shield,
    #  dia_vv_vacuum_ducts) -- folds in Vacuum.run()'s own floor(pumpn+0.5) rounding,
    # since the un-rounded pumpn has no VarPath of its own (see data footprint above).
    ...
```
All three actually written in `vacuum.py`, alongside `duct_diameter_residual`,
`duct_conductance`, `solve_duct_geometry` (the outer area-fit loop) and
`_solve_vacuum_pumping_old` (the full-diagnostic form `calculate_vacuum_pumping_old`
wraps -- see "cottax node" below for why the diagnostic form, not the public one, is
what `Tier2Contract` actually exercises).

## cottax node

**`VacuumPumpingSimple`, actually written** (`ExplicitFunction`), registered nowhere
yet (registration is the coordinating session's job per this dispatch's boundary, not
this fork's) but ready to be:
```python
class VacuumPumpingSimple(ExplicitFunction):
    n_iter_vacuum_pumps = OutputInto(vacuum)
    def __call__(self, molflow_plasma_fuelling_required=From(physics), ...):
        return calculate_vacuum_pumping_simple(...)
```
Full form in `vacuum.py`.

**No node for `calculate_vacuum_pumping_old`** (the `"old"`-branch tier-2 unit) --
genuinely blocked, not merely deferred. Three of its real inputs (`dsol`, `ritf`,
`gasld`) are composite `Vacuum.run()`-locals, each combining more than one `data`
field (see the data-footprint table above), and `cottax.interfaces.
pytree_namespace_module.FromExactly(place)` requires `place` to be a bare
attribute-access chain -- the `_Recorder` behind it rejects any arithmetic, so a
single read cannot
express `0.5*(a+b)`. Same open question `coils.md` already raised for `coilcurrent`/
`wp_width_r_min` (a minting decision, not a blocked dependency) -- not resolved here.
The full sketch (what the node would look like once `dsol`/`ritf`/`gasld` are minted as
real `VarPath`s, or the node is restructured to take `data.build`/`data.physics`
sub-namespaces and compute them inline) is left as a comment at the bottom of
`vacuum.py`, same convention as `coils.py`'s unwrapped `intersect`.

### cottax node: `DuctDiameterRootFind` (a real `RootFind`, not deferred)

Unlike `intersect`/`calculate_vacuum_pumping_old` above, `solve_duct_diameter` **does**
now get a node -- `DuctDiameterRootFind` (`vacuum.py`, an
`cottax.interfaces.pytree_namespace_module.ImplicitFunction`). This is a deliberate
reversal of `next_steps.md` §7's conclusion for this one unit, done **by explicit
instruction, not by re-deriving that finding**: §7 argued `solve_duct_diameter`'s
unknown is fully encapsulated inside `VacuumOld`'s own computation (no other node
reads or owns it), so declaring it would be structure with no consumer. That argument
is still sound as far as it goes -- nothing in this codebase reads
`DuctDiameterRootFind`'s minted `VarPath`s outside the node itself -- but this pass
was asked to build the `ImplicitFunction` anyway, as a template for the pattern
(paired with the identical, concurrently-run conversion of `coils.py`'s `intersect`),
not because a new consumer appeared. Read §7's argument as "correct but overridden
here", not as refuted.

```python
class DuctDiameterRootFind(ImplicitFunction):
    d_duct = OutputInto(vacuum)

    def residual(
        self,
        d_duct=From(vacuum),
        l1=From(vacuum),
        l2=From(vacuum),
        l3=From(vacuum),
        xmult_i=From(vacuum),
        ceff_i=From(vacuum),
    ):
        return duct_diameter_residual(d_duct, l1, l2, l3, xmult_i, ceff_i)
```

`ImplicitFunction.node_definitions` mints two nodes from this one declaration:

- the body, `NodePath(['DuctDiameterRootFind'])` -- a `CallableNode` reading all six
  `From`s above (**including** `d_duct` itself -- see below) and owning
  `^cond.vacuum.d_duct` (the residual);
- the problem, `NodePath(^problem['DuctDiameterRootFind'])` -- a `RootFind`
  `DeclaredNode` owning `.vacuum.d_duct` (the real unknown) and reading
  `^cond.vacuum.d_duct`.

Confirmed by an actual `to_graph(DuctDiameterRootFind)` call (`test_vacuum.py`'s
`test_duct_diameter_root_find_builds_cleanly`, and manually at the console): the graph
has exactly these two nodes, `g.driven` is `True`, and `g.problem_type is RootFind`.

**Every one of the six `VarPath`s here is minted, not an established `data` field** --
same precedent as `coils.py`'s `JcritIterNb3sn` (`t_helium`/`b_max`) and the
`Intersect` sketch at the bottom of that file (`wp_width_r`/`lhs`/`rhs`). None of
`d`/`l1`/`l2`/`l3`/`xmult_i`/`ceff_i` has a `data`-reachable home: all six are locals of
`_solve_vacuum_pumping_old`'s per-species loop (see the data-footprint table above --
`l1`/`l2`/`l3` are composite expressions of real fields, `xmult_i`/`ceff_i` are
per-iteration array elements). Naming choices:

- `.vacuum.d_duct` for the unknown -- a fresh name, not `.vacuum.dia_vv_vacuum_ducts`
  (the *already-established* field for the final, post-outer-loop winning diameter
  `VacuumOld` writes). This node's unknown is the per-species, per-outer-iteration
  Newton unknown -- a different quantity, at a different point in the computation, and
  reusing the established name would have been actively misleading (the residual here
  can be evaluated for any `(l1, l2, l3, xmult_i, ceff_i)`, not just the winning
  species' final call).
- `.vacuum.l1`/`.vacuum.l2`/`.vacuum.l3`/`.vacuum.xmult_i`/`.vacuum.ceff_i` -- kept
  exactly as `duct_diameter_residual`'s own parameter names, no suffix invented,
  matching `naming_convention.md`'s "port the existing name" rule and the
  `t_helium`/`b_max` precedent (kept as bare locals, not renamed for disambiguation).

**Why `d_duct` is also read as a `From` on the residual, not just declared as the
`Output`**: this is not redundant, and it's what makes the pair a genuine two-node
cycle rather than a self-loop. The `CallableNode` (the body) owns `^cond.vacuum.d_duct`
and reads `.vacuum.d_duct` -- reading a variable it does not itself own is allowed
(`spec.py`'s own rule is "a node may not read what it owns", and the body owns only the
*residual*, not the real unknown). The `RootFind` problem owns `.vacuum.d_duct` and
reads `^cond.vacuum.d_duct` -- the mirror. That two-node shape (body reads the guess,
problem owns the answer) is exactly `cottax.interfaces.pytree_namespace_module`'s own
`Disc1` example (`tests/test_interfaces_pytree_namespace.py`) and the `Sellar` fixture
it is drawn from -- not invented for this unit.

**Not registered in `total_process.py`** (out of this pass's boundary, same as every
other node in this file) and **not wired to any other node** -- `l1`/`l2`/`l3` etc.
still have no established `VarPath` upstream of them (same open question as
`calculate_vacuum_pumping_old`'s `dsol`/`ritf`/`gasld`, above), so this declaration
cannot yet be assembled into the real graph. It exists as a structural template,
verified by direct construction and a test-only driver -- see below -- not as a
production wiring.

**Test-only driver and `Drive`** (`test_vacuum.py`): `_NewtonRootFindDriver`
subclasses `cottax.evaluate.AbstractDriver` (`drives = RootFind`) and reimplements
*exactly* `solve_duct_diameter`'s own algorithm -- the same `jax.grad`-based Newton
step, the same `jax.lax.while_loop` cond/body shape, the same defaults
(`max_iter=100`, `tol=1e-10`), the same fixed `d = 1.0` start when no guess is
supplied -- just calling `conditions(d)[0]` (the block's `ConditionMap`) where
`solve_duct_diameter` calls `duct_diameter_residual(...)` directly. This is precisely
what `evaluate.py`'s own `AbstractDriver` docstring anticipates: "construct one
concrete `AbstractDriver` wrapping the exact Newton scheme this codebase already uses,
purely for test purposes... while the *structural* declaration in the real graph stays
undriven and swappable". `schedule_for(to_graph(DuctDiameterRootFind()), {problem_name:
_NewtonRootFindDriver()})` builds a real, runnable `Schedule`;
`test_duct_diameter_root_find_drive_matches_solve_duct_diameter` runs it on all 25
`_duct_diameter_samples()` points (24 fuzzed geometries + the `test_old_model` legacy
point) and checks the driven answer agrees with `solve_duct_diameter`'s own answer to
`rtol=1e-9` -- both run the identical algorithm from the identical start, so they agree
far tighter than that in practice.
`test_duct_diameter_root_find_drive_zeroes_the_residual` additionally re-plugs the
converged answer back into `duct_diameter_residual` directly, independent of the
`ConditionMap` machinery, and checks it is below `1e-8` -- the same residual-based pass
criterion `Tier2Contract` uses for `TestSolveDuctDiameter` itself.

`solve_duct_diameter` itself is **unchanged** and still what any plain caller
(`solve_duct_geometry`, `_solve_vacuum_pumping_old`) should call -- `DuctDiameterRootFind`
exists alongside it, not instead of it, the same way `duct_conductance` already sits
alongside `_newton_function`'s closed-form half.

### the outer shrink loop (`solve_duct_geometry`): kept eager, corrected framing

`solve_duct_geometry`'s outer 10%-shrink loop is **not converted** here, but the reason
changed mid-pass and the corrected framing is worth recording precisely, since the
first framing offered (a nested `Drive`-inside-something-driving-the-outer-search) was
wrong, not merely unproven.

**The right frame: `solve_duct_geometry` is a crude fixed-step stand-in for a
constrained optimisation, not a nested solve.** Reframed exactly:

```
maximise/find  ceff_i
subject to     duct_diameter_residual(d, l1, l2, l3, xmult_i, ceff_i) == 0   (RootFind --
                                                        DuctDiameterRootFind, above)
               0.25 * pi * d**2 <= a1max                (fits between adjacent TF coils)
               ceff_i >= 1.1 * s_i                       (pumping-speed floor)
```

`cottax.problem.py`'s join table states the composition directly: `Optimise.__add__`
handles `Optimise + RootFind -> Optimise` (SAND -- "the unknowns join `design`, the
residuals join `equalities`"). So the correct structure, once someone builds it, is
**one flat `Optimise` block** combining a new `Optimise` node (design variable
`ceff_i`, the two inequalities above) with `DuctDiameterRootFind`'s `RootFind` via
`+`/`Combine` -- **not** an outer driver wrapping an inner one, and not a
`FixedPoint`/nested-`Drive` shape at all. A sketch of the declaration (deliberately
**not built or wired** -- see below for what's missing) is left as a comment in
`vacuum.py`, directly above `solve_duct_geometry`.

**Why this pass stops at the sketch, not a real declaration:** real design decisions
remain open that this pass does not make and should not guess at:

1. What `a1max`/`s_i` mint to -- both are themselves composite/local quantities in
   `_solve_vacuum_pumping_old` (`a1max` depends on `rmajor`/`rminor`/`ritf`/`thcsh`/
   `n_tf_coils`; `s_i` is one element of the per-species required-speed array `s`),
   same open-question shape as `dsol`/`ritf`/`gasld` above.
2. What the `Optimise` problem's `objective` actually is. "Largest feasible `ceff_i`"
   is not yet posed as a scalar to minimise/maximise -- `objective = -ceff_i` is the
   obvious candidate, but PROCESS's own loop doesn't actually *maximise* `ceff_i`
   against the constraints in one shot; it re-solves the `RootFind` at each of a
   sequence of shrunk `ceff_i` values and stops at the first that fits, which is one
   particular (crude, non-optimal) algorithm for approximately solving this
   optimisation, not evidence for what the "right" objective is. Getting this wrong
   would silently change what PROCESS's own duct-sizing model is claimed to compute.
3. How the two inequality residuals (`0.25*pi*d**2 - a1max <= 0`,
   `1.1*s_i - ceff_i <= 0`) get their own `Compare`/`^cond` nodes -- mechanical once
   (1)/(2) are settled, not attempted here.

None of these is a "might be hard" hedge -- each is a concrete open question with a
concrete consequence for getting it wrong, which is why this pass declares the shape
and stops rather than guessing values for any of the three. `solve_duct_geometry` and
`_solve_vacuum_pumping_old` are unchanged, still eager, and still what
`TestVacuumPumpingOld`/`TestVacuumPumpingOldFromFields` test today.

## tier signal

- `calculate_vacuum_pumping_simple`: **tier 1.** No iteration, no calls into other
  models, no switches (see switches table -- none touched).
- `solve_duct_diameter`/`duct_diameter_residual`: **tier 2, self-contained.** A genuine
  internal Newton-Raphson solve (`Vacuum._newton_method_duct_diameter`'s inner loop),
  no calls into any not-yet-ported unit. Same shape as `coils.py`'s `intersect`,
  ported the same way (real, convergence-checked driver replacing PROCESS's fixed-
  iteration loop, `Tier2Contract`'s residual-based pass criterion). **Additionally**
  given a genuine `ImplicitFunction`/`RootFind` structural declaration
  (`DuctDiameterRootFind`), unlike `intersect` -- see "cottax node:
  `DuctDiameterRootFind`" above for why this one unit's treatment diverges from
  `next_steps.md` §7's general conclusion.
- `_solve_vacuum_pumping_old`/`calculate_vacuum_pumping_old`: **tier 2, self-contained.**
  Composes the above with an outer area-fit loop
  (`_newton_method_duct_diameter`'s outer `while True`, ported as `solve_duct_geometry`)
  and a 4-species sequential decision loop (`vacuum()`'s own `for i in range(4)`,
  which threads `imax`/`cmax`/`pumpn` state across iterations via `jax.lax.fori_loop`
  + `jax.lax.cond`). No calls into any not-yet-ported unit -- every read is either an
  explicit argument or one of the `.vacuum.*`/`.times.*` fields tabulated above, all of
  which are plain data, not another model's output. `Vacuum.run()`'s own rounding step
  is folded in (see proposed signature).

## switches touched

- `.vacuum.i_vacuum_pumping` (`"old"`/`"simple"`) -- **topology-changing**, resolved at
  `Vacuum.run()`. New entry, not yet in `_audit/units/core/solver/switches.md` (out of
  this fork's edit boundary; flagging for the coordinating session). The two branches'
  write-sets are essentially disjoint (`"old"`: `n_vv_vacuum_ducts`/`dlscal`/
  `m_vv_vacuum_duct_shield`/`dia_vv_vacuum_ducts`/`n_vac_pumps_high`; `"simple"`:
  `n_iter_vacuum_pumps` only) -- a clean **split** by the reads/writes-set evidence,
  same recommendation `switches.md`'s existing topology entries (`isthtr`,
  `ipowerflow`) already follow.
- `.vacuum.i_vacuum_pump_type` (`VacuumPumpType`, 0 turbomolecular / 1 compound
  cryopump) -- **formula-changing, same reads-set both values** (only the pump-speed
  table `sp` and the final `pumpn *= 2` multiplier differ; every other read is
  identical). Per `naming_convention.md` this is a "static kwarg" candidate, but this
  port implements it with `jnp.where` over both candidate `sp` tables instead (see
  `_solve_vacuum_pumping_old`'s body) -- a deliberate deviation from the
  static-kwarg convention, made because `Tier2Contract` never differentiates `ported`
  (so there's no autodiff cost to keeping it traced) and it avoids a second design
  decision about whether the value arrives as a Python `int` or a traced array at the
  node boundary. Flagging this as a precedent inconsistency for whoever writes
  `switches.md`'s entry.
- `.vacuum.i_vac_pump_dwell` (0/1/2) -- **formula-changing, but data-entangled**: the
  branch that picks `tpump`'s formula is `i_vac_pump_dwell == 1 or t_plant_pulse_dwell
  == 0`, so which formula applies depends on both the switch *and* the runtime value of
  a continuous input -- not resolvable as a pure compile-time switch at all. Ported as
  a `jnp.where` chain (see `_solve_vacuum_pumping_old`'s body), not a static kwarg.
  Genuinely new shape, not covered by `naming_convention.md`'s two-case split
  ("topology-changing" / "formula-changing with identical reads-set") -- flagging as an
  open question for that document, not resolved here.

## calls into other models

None. Every read in this file is either an explicit argument, a `.vacuum.*`/
`.times.*`/`.physics.*`/`.build.*`/`.tfcoil.*`/`.divertor.*` plain-data field, or (for
`VacuumVessel`, out of scope) `process.models.build.FwBlktVVShape` /
`process.models.engineering.ivc_functions`.

## JAX-difficulty flags

- **`_newton_method_duct_diameter`'s inner Newton loop** (`workaround-known`): fixed
  `d = 1.0` start, up to 100 steps, data-dependent early `break` on `dd <= 0.01`. No
  faithful fixed-trip-count translation (JAX has no early exit); ported as
  `jax.lax.while_loop` in `solve_duct_diameter` instead. `Tier2Contract` never
  differentiates `ported` (see `_harness/contracts.py`'s docstring), so
  `while_loop`'s lack of autodiff support costs nothing here -- worth flagging for
  whoever eventually wants `jacfwd` through this unit (e.g. for a tier-4 MDA), since
  `while_loop` would need replacing with a bounded, `jnp.where`-frozen
  `lax.fori_loop` at that point, the same tradeoff `intersect`'s `optimistix`-based
  design already made in the other direction.
- **`_newton_method_duct_diameter`'s outer area-fit loop** (`workaround-known`): a
  genuinely unbounded `while True` in the source (shrink `ceff[i]` by 10% until it
  fits or gives up) -- ported as `jax.lax.while_loop` in `solve_duct_geometry`, same
  reasoning as above, with a documented `max_outer=64` safety cap (see that function's
  docstring; `0.9**64 ~ 1.2e-3`, far past where any physically plausible input would
  have already exited).
- **`vacuum()`'s 4-species `for i in range(4)` loop** (`workaround-known`): sequential,
  stateful (`imax`/`cmax`/`pumpn` carried across iterations, `ceff`/`d` arrays
  accumulated), with a Python `continue`/process branch per species. Ported as
  `jax.lax.fori_loop` (fixed 4 trips, no data-dependent count -- straightforward) with
  `jax.lax.cond` for the branch and `.at[i].set(...)` for the accumulator writes.
  Downstream indexing by the *data-dependent* winning species (`d[imax]`,
  `ceff[imax]`, `xmult[imax]`) uses `jnp.take`/plain `[]` indexing with a traced
  integer, which XLA lowers to a dynamic-slice/gather -- no difficulty beyond the
  ordinary JAX idiom.
- `jax.grad(duct_diameter_residual)` used inside `solve_duct_diameter`'s own
  `lax.while_loop` body (`minor`): nested autodiff-inside-a-loop-body is fine in JAX
  (the inner `grad` traces a plain closed-form expression, no loops of its own) --
  noted only because it replaces PROCESS's hand-derived analytic derivative
  (`_newton_function`'s `dy`) with an autodiff one; verified to agree with PROCESS's
  own formula to full float64 precision at the same point (see "Real findings" #4).

## open questions

1. **`dsol`/`ritf`/`gasld` minting** (see "cottax node" above) -- blocks
   `calculate_vacuum_pumping_old`'s node. Same shape as `coilcurrent`/`wp_width_r_min`
   in `coils.md`, not resolved there either; a real design decision for whoever owns
   `VarPath`-minting policy, not this fork.
2. **`max_outer=64`'s cap** (`solve_duct_geometry`) is a safety bound PROCESS's own
   code does not have (a genuine unbounded `while True`). No fuzz sample in
   `test_vacuum.py` came close to needing more than a handful of shrink steps, but a
   pathological input (TF coil geometry that makes `a1max` tiny or negative relative to
   any achievable duct area) could in principle need more than 64 -- in that regime the
   loop would exit still "not done", `nflag` frozen at whatever it last was and
   `ceff_final` partway through its shrink sequence, silently different from what an
   unbounded loop would produce. Not exercised or resolved here; flagging as a known
   edge PROCESS's own regression inputs are not expected to reach (its documented
   real-world use is nowhere near this regime), rather than proven unreachable.
3. **Fuzz coverage deliberately avoids the `nflag = 1` ("space limited") regime.**
   `_vacuum_pumping_old_samples()`'s bounds are centred on `test_old_model`'s own
   scale, chosen so `a1max` comfortably exceeds any of the four species' converged
   duct area in every sample drawn (verified: no fuzz point in the current suite
   triggers a shrink-loop iteration at all, `imax`'s Newton solve always fits on its
   first attempt). This is a real coverage gap for real finding #6 above (the
   `ceff[imax]`/`d[imax]` inconsistency only shows up in that regime) -- not exercised
   because constructing physically-plausible geometry that reliably trips it (rather
   than by chance) needs a deliberate, separate sample construction this fork did not
   attempt.
4. **`switches.md` has no case yet for a "formula-changing but data-entangled" switch**
   (`.vacuum.i_vac_pump_dwell`, see switches table) -- `naming_convention.md`'s two-way
   split (topology-changing / formula-changing-with-identical-reads-set) doesn't
   describe a branch whose condition mixes a switch value with a continuous runtime
   comparison. Flagging for whoever next revises that document, not resolved here.
5. **`solve_duct_geometry`'s `Optimise` declaration** (see "the outer shrink loop"
   above) -- the shape is settled (`Optimise + RootFind -> Optimise`, per
   `cottax.problem`'s join table), but three real decisions are open before it can be
   built: what `a1max`/`s_i` mint to, what the objective actually is (PROCESS's own
   10%-shrink loop is one crude algorithm for approximately solving this, not evidence
   for the "correct" objective), and how the two inequalities become `Compare`/`^cond`
   nodes. Left as a sketch-only comment in `vacuum.py`, not a working declaration --
   whoever picks this up should treat the objective question especially carefully,
   since getting it wrong would silently misrepresent what PROCESS's duct-sizing model
   claims to optimise.

## tokamak-scope addendum (wave-1 dispatch)

Added by a later pass porting `.tokamak.vacuum_vessel`, confirming this record's own
earlier prediction (§ source, above: "`VacuumVessel` ... is out of scope" — true only
for the stellarator, as flagged there at the time). `Vacuum` (everything above this
section) is untouched.

### source

`process/models/vacuum.py:736-994`, `VacuumVessel` class. Reached on the tokamak path
at `caller.py:331` (`unit #16`'s own note: "confirmed unreachable on the stellarator
pipeline, no action needed" — the tokamak trace is the confirmation the other half of
that sentence anticipated). In scope: `VacuumVessel.run()` (742-799),
`calculate_vessel_half_height` (801-855, `@staticmethod`),
`calculate_elliptical_vessel_volumes` (899-933, `@staticmethod`). Out of scope:
`calculate_dshaped_vessel_volumes` (857-897, `@staticmethod`, D-shaped arm, UNPORTED —
see § switches), `output()` (975-994, pure reporting).

### data footprint

| VarPath | read/write | classification | note |
|---|---|---|---|
| `.build.z_tf_inside_half` | read | explicit-arg | `:745` |
| `.build.dz_shld_vv_gap` | read | explicit-arg | `:746` |
| `.build.dz_vv_lower` | read | explicit-arg | `:747` |
| `.divertor.n_divertors` | read | switch (branch) | `:748` — see § switches; not a port parameter |
| `.build.dz_blkt_upper` | read | explicit-arg | `:749` — single-null arm only |
| `.build.dz_shld_upper` | read | explicit-arg | `:750` — same arm |
| `.build.z_plasma_xpoint_upper` | read | explicit-arg | `:751` — same arm |
| `.build.dr_fw_plasma_gap_inboard` | read | explicit-arg | `:752` — same arm |
| `.build.dr_fw_plasma_gap_outboard` | read | explicit-arg | `:753` — same arm |
| `.build.dr_fw_inboard` | read | explicit-arg | `:754` — same arm |
| `.build.dr_fw_outboard` | read | explicit-arg | `:755` — same arm |
| `.blanket.dz_vv_half` | write | local-intermediate | `:744`, read back at `:769,786` (same straight-line call) — kept a local in the port; `structure.md`'s `dewmkg` cross-reference note applies in reverse here (nothing outside this chain reads `dz_vv_half` in `process/`, confirmed by grep) |
| `.physics.itart` | read | switch | `:759` — see § switches |
| `.fwbs.i_fw_blkt_vv_shape` | read | switch | `:760` — see § switches |
| `.physics.rmajor` | read | explicit-arg | `:781` |
| `.physics.rminor` | read | explicit-arg | `:781` |
| `.physics.triang` | read | explicit-arg | `:783` — elliptical arm only |
| `.build.r_shld_inboard_inner` | read | explicit-arg | `:767,785` |
| `.build.r_shld_outboard_outer` | read | explicit-arg | `:768,785` |
| `.build.dr_vv_inboard` | read | explicit-arg | `:770,787` |
| `.build.dr_vv_outboard` | read | explicit-arg | `:771,788` |
| `.build.dz_vv_upper` | read | explicit-arg | `:772,789` |
| `.blanket.vol_vv_inboard` | write | explicit-arg | `:762-779` |
| `.blanket.vol_vv_outboard` | write | explicit-arg | same |
| `.fwbs.fvoldw` | read | explicit-arg | `:796` |
| `.fwbs.vol_vv` | write (twice) | redundant-duplicate-write-adjacent | `:765/779` write the raw shell volume, `:796` overwrites with the coverage-scaled one — the port keeps only the final value (`calculate_vacuum_vessel_mass`'s first return), matching PROCESS's own net effect |
| `.fwbs.den_steel` | read | explicit-arg | `:799` |
| `.fwbs.m_vv` | write | explicit-arg | `:799` — `tokamak_boundary.md`'s one declared read of this slot |

### proposed signature(s)

```python
def calculate_vessel_half_height(
    z_tf_inside_half, dz_shld_vv_gap, dz_vv_lower, dz_blkt_upper, dz_shld_upper,
    z_plasma_xpoint_upper, dr_fw_plasma_gap_inboard, dr_fw_plasma_gap_outboard,
    dr_fw_inboard, dr_fw_outboard,
) -> float:  # dz_vv_half, n_divertors == 1 baked in

def calculate_elliptical_vessel_volumes(
    rmajor, rminor, triang, r_shld_inboard_inner, r_shld_outboard_outer, dz_vv_half,
    dr_vv_inboard, dr_vv_outboard, dz_vv_upper, dz_vv_lower,
) -> tuple[float, float, float]:  # vol_vv_inboard, vol_vv_outboard, vol_vv (pre-coverage)

def calculate_vacuum_vessel_mass(vol_vv_raw, fvoldw, den_steel) -> tuple[float, float]:  # vol_vv, m_vv

def calculate_vacuum_vessel_outputs(...) -> tuple[float, float, float, float, float]:
    # composes the three above, mirrors VacuumVessel.run() end to end
```

### cottax node

`VacuumVesselElliptical(ExplicitFunction)`, in `functional_process/models/vacuum/
vacuum.py`, appended after `VacuumOld`. Owns `dz_vv_half`, `vol_vv_inboard`,
`vol_vv_outboard`, `vol_vv`, `m_vv`. Proposed as the sole occupant of
`.tokamak.vacuum_vessel` (registration is the consolidation pass's job).

### tier signal

**Tier 1**, all three functions (plus the composite). No iteration, no calls into
another model's method, no CoolProp.

**Sample provenance.** `tests/unit/models/test_vacuum.py::test_elliptical_vessel_
volumes` provides one legacy point for `calculate_elliptical_vessel_volumes`, used
verbatim. No legacy oracle exists for `calculate_vessel_half_height` or the composite
`VacuumVessel.run()` pipeline; both are fuzz-only.

### switches touched

| switch | reachable values | live on `large_tokamak_eval` | decision | evidence |
|---|---|---|---|---|
| `.divertor.n_divertors` | `1`, `2` | `1` — derived by `process/core/init.py:606-616` from `.physics.i_single_null = 1` (`large_tokamak_eval.IN.DAT:307`), not the field's own default of `2` | **split** (baked, no parameter) | `calculate_vessel_half_height`'s `if n_divertors == 2: z_top = z_bottom else: ...` (`vacuum.py:845-851`) reads a materially different set of fields per arm. Same switch, same live value, as `structure.md`/`divertor.md`/`fw.md` |
| `.physics.itart` / `.fwbs.i_fw_blkt_vv_shape` (compound) | `itart ∈ {0,1}`, `i_fw_blkt_vv_shape ∈ {D_SHAPED=1, ELLIPTICAL_SHAPED=2}` | `itart = 0`, `i_fw_blkt_vv_shape = 2` (both PROCESS defaults) → elliptical arm | **split** (baked, no parameter) | `vacuum.py:758-791`: D-shaped arm reads no `triang`; elliptical arm does. Same compound switch, resolved the same way, as `fw.md`'s analogous first-wall-area dispatch — independently confirmed rather than assumed shared |

### calls into other models

None.

### JAX-difficulty flags

None beyond `ivc_functions.md`'s own (division by strictly-positive shell half-widths
inside `eshellvol`). No fractional powers, no CoolProp, no in-place mutation.

### deviations from PROCESS

- **`.fwbs.vol_vv`'s intermediate write is dropped** — PROCESS writes it once with the
  raw shell volume (`:765` or `:779`, depending on branch) and again with the
  `fvoldw`-scaled value (`:796`); only the final value is a real output (the first
  write is immediately superseded within the same call, `redundant-duplicate-write` in
  spirit though not literally the same value twice). The port's
  `calculate_vacuum_vessel_mass` produces only the final value.

### open questions (addendum)

- **Registration** — same open question as `fw.md`/`divertor.md`: `VacuumVesselElliptical`
  is proposed as `.tokamak.vacuum_vessel`'s sole occupant, left to the consolidation pass.
- **D-shaped arm** — UNPORTED. A future occupant for `itart == 1 or i_fw_blkt_vv_shape ==
  D_SHAPED` would call the already-source-side `calculate_dshaped_vessel_volumes`
  (`dshellvol`-based); `functional_process/models/engineering/ivc_functions.py` would
  need `dshellvol` added at that point (currently only the elliptical pair is ported —
  see that module's own docstring).


## 2026-08-27 — the `n_divertors == 2` arm ported (double-null wave)

`.tokamak.vacuum_vessel` is a family of two now: `VacuumVesselElliptical` is the abstract
base — "Elliptical" naming the shape switch both members still bake in —
`VacuumVesselEllipticalSingleNull` the occupant this slot already had (unchanged in
body), and `VacuumVesselEllipticalDoubleNull` the new one. `indat.py`'s `VACUUM_VESSEL`
registry gains arm `1` and the `('vacuum_vessel_arm', -1)` refusal is gone.

**Half-height (`vacuum.py:843-851`).** `z_top = z_bottom`, so the answer is
`z_tf_inside_half - dz_shld_vv_gap - dz_vv_lower` — written reduced, as in `shield.py`,
`blankets/blanket_library.py` and `models/fw.py`.

**Seven fewer reads, the largest arm difference in this wave.** The `else` arm builds
`z_top` from `z_plasma_xpoint_upper`, `dr_fw_plasma_gap_inboard`/`_outboard`,
`dr_fw_inboard`/`_outboard`, `dz_blkt_upper` and `dz_shld_upper`; a double-null vessel
reads none of them. `dz_vv_lower` survives into the double-null composite only because
the *volume* call reads it too. Seven invented edges is what a `jnp.where` over one node
would have cost here — the clearest single illustration in this wave of why an arm that
differs in reads is an occupant rather than a parameter.

**Tests.** `TestCalculateVesselHalfHeightDoubleNull` and
`TestCalculateVacuumVesselOutputsDoubleNull`, both Tier 1, over the single-null
contracts' own boxes minus the seven dropped inputs. Both adapters **poison** all seven
with `nan` rather than zeroing them; the composite can do it because
`process/models/vacuum.py` reads every one of them in exactly one place (`:744-756`, the
arguments of the half-height call), so on this arm nothing may touch them and a `nan`
proves it rather than a zero hiding it. Green at `--fp-gradients --fp-fuzz 40`.

`_vacuum_vessel_arm` was reordered to ask the shape first, for the same reason
`_first_wall_arm` was — see `fw.md`. The two spherical-tokamak files now stop at
`('fw_blkt_vv_shape_arm', 0)`.

No new boundary input. The § Registration open question is unchanged in substance; it now
concerns a family of two rather than a lone class.

## 2026-08-27 — the D-shaped double-null cell ported (D-shaped wave)

`.tokamak.vacuum_vessel` is the second of the two slots whose occupant grid is a shape ×
divertor-count **product** — see `fw.md`'s dated section for the general point, which
this unit confirms independently for the third time (wave 1 for the predicate itself, the
double-null wave for the divertor axis, this wave for the product).

**What was added.** `calculate_dshaped_vessel_volumes` (ports
`process/models/vacuum.py:860-906`, verbatim, calling the new shared
`ivc_functions.dshellvol`), the composite
`calculate_vacuum_vessel_outputs_dshaped_double_null`, and the occupant
`VacuumVesselDShapedDoubleNull`. The D-shaped single-null cell is refused at
`('vacuum_vessel_arm', -2)` for the same reachability reason as its first-wall twin.

**The sharpest reads-set difference in this wave.** The D-shaped volume formula anchors on
the shield build alone (`r_1 = r_shld_inboard_inner`, `r_2 = r_shld_outboard_outer - r_1`)
where the elliptical one centres its ellipses on the plasma
(`r_1 = rmajor - rminor * triang`). So it drops `.physics.rmajor`, `.physics.rminor` **and**
`.physics.triang` — and since the double-null half-height had already dropped its seven,
`VacuumVesselDShapedDoubleNull` reads **ten** fields against the elliptical single-null
occupant's twenty, with **no `From(physics)` port at all**. An entire namespace edge that
the elliptical arms declare and this one does not: the clearest single demonstration in
the port so far of what the union-of-arms reads-set costs.

**Tests.** `TestCalculateDshapedVesselVolumes` (bare PROCESS staticmethod, no adapter and
no poisoning possible — the signature has no `.physics` parameters) and
`TestCalculateVacuumVesselOutputsDshapedDoubleNull` (real `VacuumVessel.run()`, adapter
poisons **ten** fields with `nan`: the double-null arm's seven plus the three `.physics`
ones, read at `vacuum.py:781-783` only, inside the elliptical volume call). Both green at
`--fp-gradients`.

One fuzz-box note. `r_shld_inboard_inner` starts at `1.5` on the D-shaped contracts where
the elliptical ones use `0.5`: on this arm it *is* `dshellvol`'s `rmajor`, whose inboard
term `rmajor**2 - (rmajor - drin)**2` needs `drin < rmajor`, and `dr_vv_inboard` runs to
`1.0`. On the elliptical arm the same field is one term of a difference and no such
constraint applies. PROCESS has no guard either way, so both sides would agree on the
nonsense — the bound keeps draws physical rather than hiding a disagreement.

`_vacuum_vessel_arm` gained a third written arm and its refusal narrowed from "the
D-shaped vessel" to "the D-shaped vessel at a single divertor". No new boundary input.
