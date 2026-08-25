---
kind: model-unit
status: draft
confidence: high
---

**Ported (2 units).** `power_A_tf_coil_power.py` / `test_power_A_tf_coil_power.py`:
`calculate_tf_power_resistive` and `calculate_tf_power_superconducting`, both tier-1.
Registered nowhere yet -- registration/`total_process.py` wiring is a later
consolidation pass, out of this fork's edit boundary (see the top-level task
instructions).

## source

`process/models/power.py` (registry unit #14), chunk A of 3 (see
`power_B_thermal_cryo.md`/`power_C_electric_production.md` for the other two --
split because the six requested methods span three largely-independent sub-domains
of the same file: TF coil power supply sizing (this chunk), thermal power balance +
cryogenics, and AC/electric production over the pulse. Same rationale
`physics_A/B/C.md` used at comparable size).

- `Power.tfpwr` (2117-2287) -- dispatches on `.tfcoil.i_tf_sup` (`!= 1` resistive /
  `== 1` superconducting) to one of two essentially disjoint computations, same shape
  as `vacuum.py`'s `.vacuum.i_vacuum_pumping` dispatch. Reporting section (`if output:
  ...`, 2210-2289) excluded from scope.
- `Power.tfpwcall` (2291-2330) -- the superconducting branch's shell: computes
  `ettfmj`/`itfka` and calls `tfcpwr`. No `VarPath` of its own beyond what it forwards.
- `Power.tfcpwr` (2332-2629) -- the actual superconducting power-conversion-system
  sizing. Reporting section (`if output: ...`, 2495-2627) excluded from scope.

## data footprint

### `Power.tfpwr`, `i_tf_sup != 1` branch (resistive)

| VarPath | read/write | classification | note |
|---|---|---|---|
| `.tfcoil.i_tf_sup` | read | topology-changing switch | resolved at graph-assembly time, not a port -- see switches below |
| `.tfcoil.c_tf_turn`, `.tfcoil.j_tf_bus`, `.tfcoil.rho_tf_bus`, `.tfcoil.len_tf_bus`, `.tfcoil.n_tf_coils`, `.tfcoil.res_tf_leg`, `.tfcoil.p_cp_resistive`, `.tfcoil.c_tf_total`, `.tfcoil.p_tf_joints_resistive`, `.tfcoil.p_tf_leg_resistive` | read | explicit-arg | |
| `.heat_transport.etatf` | read | explicit-arg | |
| `.tfcoil.m_tf_bus`, `.tfcoil.vtfkv`, `.tfcoil.p_cp_resistive_mw`, `.tfcoil.p_tf_leg_resistive_mw`, `.tfcoil.p_tf_joints_resistive_mw`, `.tfcoil.tfcmw` | write | explicit-arg | |
| `.heat_transport.p_tf_electric_supplies_mw` | write | explicit-arg | also written by the superconducting branch (different formula) -- the two branches never both run in one call, so not a real double-write |

`tfbusres`, `res_tf_system_total`, `tfbusmw`, `tfreacmw` are locals with no `VarPath`
(the first two feed the output section only, out of scope; `tfreacmw` is a hardcoded
`0.0`, see real findings). Dropped from the port's return.

### `Power.tfpwcall` / `Power.tfcpwr`, `i_tf_sup == 1` branch (superconducting)

| VarPath | read/write | classification | note |
|---|---|---|---|
| `.tfcoil.e_tf_magnetic_stored_total_gj`, `.tfcoil.n_tf_coils`, `.tfcoil.c_tf_turn` | read | explicit-arg | feed `tfpwcall`'s locals `ettfmj`/`itfka`, then passed straight into `tfcpwr` |
| `.physics.rmajor`, `.tfcoil.v_tf_coil_dump_quench_kv`, `.tfcoil.res_tf_leg` | read | explicit-arg | `tfcpwr`'s own parameters (`rmajor`, `v_tf_coil_dump_quench_kv`, `rptfc`) |
| `.tfcoil.rho_tf_bus` | read | explicit-arg | only `self.data.*` read inside `tfcpwr`'s body itself (line 2402), everything else is a parameter |
| `.heat_transport.etatf` | read | explicit-arg | only other `self.data.*` read inside `tfcpwr`'s body (line 2486) |
| `.tfcoil.tfckw`, `.tfcoil.len_tf_bus`, `.tfcoil.drarea` | write | explicit-arg | `tfcpwr`'s tuple return, unpacked by `tfpwcall` |
| `.buildings.tfcbv` | write | explicit-arg | same tuple, written into a *different* area (`buildings`, not `tfcoil`) -- a real cross-area write, not unusual for this codebase but worth flagging: this is the field `buildings.py` (unit #15) presumably consumes as an input for building sizing |
| `.heat_transport.p_tf_electric_supplies_mw` | write | explicit-arg | same field the resistive branch writes, different formula -- see above |

`.tfcoil.len_tf_bus` is **read** by the resistive branch and **written** by the
superconducting branch -- confirmed the two branches never overlap (mutually
exclusive on `i_tf_sup`), so this is not a real read/write race, just PROCESS reusing
one field's storage for two different roles depending on TF coil type.

Every other `tfcpwr` local (`ncpbkr`, `djmka`, `rtfps`, `fspc1-3`, `tchghr`,
`nsptfc`, `ettfc`, `ltfth`, `ntfbkr`, `lptfcs`, `albusa`, `albuswt`, `rtfbus`,
`vtfbus`, `rcoils`, `ztotal`, `tfcv`, `ntfpm`, `tfpmv`, `tfpsv`, `tfpska`, `tfpmka`,
`tfpmkw`, `tfackw`, `r1dump`, `ttfsec`, `ndumpr`, `r1ppmw`, `r1emj`, `rpower`,
`xpower`, `part1-3`, `tfcfsp`, `xpwrmw`) is a same-function local with no `VarPath` --
either an intermediate feeding one of the five returns, or feeding only the
out-of-scope output section (`albuswt`, `vtfbus`, `tfpsv`, `tfpska`, `ttfsec`,
`r1ppmw`, `xpower`, `xpwrmw`, `lptfcs`, `r1dump` are display-only and dropped from the
port's computation entirely, same convention as `vacuum.py`'s pruning).

No `implicit-io`, `implicit-io-via-callee` or `redundant-duplicate-write` anywhere in
this chunk.

## Real findings (documented, not fixed)

1. **`tfreacmw` (TF coil reactive power) is a hardcoded `0.0`** in the resistive
   branch ("Set reactive power to 0, since ramp up can be long... has been removed
   (#199 #847)", `power.py:2181-2189`). Not a bug -- an intentional simplification,
   noted here only because the port drops the dead `+ tfreacmw` term rather than
   carrying a `+ 0.0` for fidelity.
2. **`tfcpwr`'s own notion of "superconducting" (`rptfc == 0.0`, selecting `nsptfc`/
   `tchghr`) is independent of, and can disagree with, `tfpwr`'s `i_tf_sup == 1`
   dispatch that decided to call `tfcpwr` at all.** `tfcpwr` is only reached when
   `i_tf_sup == 1`, but nothing forces `.tfcoil.res_tf_leg` to be exactly `0.0` in
   that case -- a real superconducting TF coil generally has *some* small nonzero
   leg/joint resistance. In practice `tfcpwr`'s "resistive" sub-case (`nsptfc = 0`,
   10-minute charge time) is therefore the one actually exercised whenever `i_tf_sup
   == 1` in realistic inputs, and the `rptfc == 0.0` sub-case models an idealised
   limit more than a realistic operating point. Not a bug -- two independently-named
   "superconducting" concepts in the same file, worth knowing before assuming
   `nsptfc` tracks `i_tf_sup`.

## proposed signature(s)

```python
def calculate_tf_power_resistive(
    c_tf_turn,
    j_tf_bus,
    rho_tf_bus,
    len_tf_bus,
    n_tf_coils,
    res_tf_leg,
    p_cp_resistive,
    c_tf_total,
    p_tf_joints_resistive,
    p_tf_leg_resistive,
    etatf,
) -> tuple[float, float, float, float, float, float, float]:
    # (m_tf_bus, vtfkv, p_cp_resistive_mw, p_tf_leg_resistive_mw,
    #  p_tf_joints_resistive_mw, tfcmw, p_tf_electric_supplies_mw)
    ...


def calculate_tf_power_superconducting(
    c_tf_turn,
    e_tf_magnetic_stored_total_gj,
    n_tf_coils,
    rmajor,
    v_tf_coil_dump_quench_kv,
    res_tf_leg,
    rho_tf_bus,
    etatf,
) -> tuple[float, float, float, float, float]:
    # (tfckw, len_tf_bus, drarea, tfcbv, p_tf_electric_supplies_mw)
    ...
```
Both actually written in `power_A_tf_coil_power.py`.

## cottax node

`TfPowerResistive` and `TfPowerSuperconducting`, both `ExplicitFunction`, actually
written in `power_A_tf_coil_power.py`. Neither declares `.tfcoil.i_tf_sup` as a
`From` read or a static field -- per `_audit/naming_convention.md`, a topology-changing
switch is consumed by the code that assembles the graph (choosing which of the two
nodes to instantiate), not represented on either node. Same convention as `vacuum.py`'s
`VacuumPumpingSimple`/`calculate_vacuum_pumping_old` split.

## tier signal

Both **tier 1**: no internal iteration, no calls into any other model, no `self.data`
access once ported. `calculate_tf_power_superconducting` has one internal branch
(`res_tf_leg == 0.0`, see switches below) but it is a plain algebraic selection
(`jnp.where`), not an iterative solve.

## switches touched

- `.tfcoil.i_tf_sup` (0 resistive / 1 superconducting / 2 aluminium, but `tfpwr` only
  distinguishes `== 1` vs `!= 1`) -- **topology-changing**, resolved at `Power.tfpwr`.
  New entry, not yet in `_audit/units/core/solver/switches.md` (out of this fork's edit
  boundary, flagging for the coordinating session). Write-sets are disjoint except for
  the shared `.heat_transport.p_tf_electric_supplies_mw` (different formula each
  branch, both write it) -- a clean **split**, same recommendation as
  `vacuum.md`'s `i_vacuum_pumping`.
- `res_tf_leg == 0.0` inside `tfcpwr` (PROCESS's own `rptfc == 0.0e0` test,
  `# noqa: RUF069` at `power.py:2365`) -- **not a `data` switch**, a runtime equality
  test on a continuous physical quantity. Ported as `jnp.where`, differentiable
  everywhere except the exact singular point itself -- see JAX-difficulty flags.

## calls into other models

None. Every read is either an explicit argument or a `.tfcoil.*`/`.heat_transport.*`/
`.physics.*` plain-data field; every write lands in `.tfcoil.*`/`.heat_transport.*`/
`.buildings.*` (the last presumably consumed by `buildings.py`, unit #15/16, already
ported per the concurrent consolidation pass -- confirmed by reading
`functional_process/models/buildings.py` directly, not by trusting
`unit_registry.md`'s status column, per this task's instructions; this port does not
call into it, it only writes a `VarPath` `buildings.py` may read).

## JAX-difficulty flags

- **`jnp.sqrt(n_tf_coils * res_tf_leg * 1000.0)` at `res_tf_leg = 0.0`** (`minor`,
  not reproducible with a guard): the JVP rule for `sqrt` divides by
  `2 * sqrt(primal)`; at `res_tf_leg = 0.0` that is `0 * inf = NaN` for *every*
  differentiated argument's column, not just `res_tf_leg`'s own -- because the local
  derivative itself is infinite there, independent of which direction's tangent is
  flowing through it. Verified this is not a `jnp.where`-hides-a-bug artifact (the
  category `physics_A_pure_formulas.md` names): the *value* at `res_tf_leg = 0.0`
  agrees with PROCESS to float64 round-off (checked by hand, not through the
  harness -- see open questions), and the source formula's own analytic derivative
  really is unbounded there (a genuine square-root singularity of `sqrt(k*x)` at
  `x = 0`), not a spurious NaN a differently-written guard would remove. `res_tf_leg`
  is also not physically expected to be exactly zero in realistic inputs (see real
  finding #2) -- `test_power_A_tf_coil_power.py`'s fuzz bounds deliberately exclude
  it, same policy `vacuum.md` used for its own unexercised edge (open question #3
  there).

## open questions

1. **`res_tf_leg = 0.0`'s value agreement was checked by hand** (a one-off REPL
   comparison against `Power.tfpwr`, matching to float64 round-off), not added as a
   harness sample, because doing so would fail `test_gradient_finite`/
   `test_gradient_agreement` for a reason that has nothing to do with a porting bug
   (see JAX-difficulty flags). If `_harness/contracts.py` ever grows a per-sample
   gradient-test opt-out, this point is a good candidate to add back for permanent
   value-level regression coverage.

## Derivative-safe power laws (`safe_pow` / `safe_sqrt`)

4 fractional power laws and 1 square root in this file have been rewritten from `x ** p` / `jnp.sqrt(x)` to
`models/safe_math.py`'s `safe_pow(x, p)` / `safe_sqrt(x)`.

**Why.** For `0 < p < 1` the function is continuous at `x == 0` and its derivative is
not: `d/dx x**p = p * x**(p-1) -> +inf`. JAX's JVP then returns `inf` along the
direction that perturbs `x` and `nan` (`inf * 0`) along every other, so the *value* is
right everywhere and the *Jacobian row* is poisoned. That is the defect class
`_audit/next_steps.md` §9 records; the most recent instance produced 46 non-finite
Jacobian cells and stalled a cold optimiser start at zero SQP steps, reported by the
solver as "the problem seems to be non-convex".

**Value identity, checked not asserted.** `safe_pow`/`safe_sqrt` dispatch on `x == 0`
and evaluate the identical expression otherwise, so every `x != 0` result is bit-for-bit
what it was, and the `x == 0` result is `0.0 ** p` / `sqrt(0.0)` -- again exactly what
the bare expression returns. Verified two ways: a hex-exact diff of every Tier-1
contract's output over every declared sample plus eight fresh fuzz draws (3655 points,
zero differing bits), and `run_mda_harness.py` unchanged at 492 agreements / 34
disagreements. PROCESS itself does not raise at `x == 0` here -- it is plain Python
`float.__pow__` / `numpy.sqrt`, both of which return `0.0` -- and the reference was
re-evaluated at each boundary point to confirm it returns the port's number.

**What changed is only the derivative at exactly `x == 0`**, which becomes `0` instead
of `inf`/`nan` -- the same convention JAX already uses at `jnp.maximum`'s kink.

`Tier1Contract.test_gradient_finite_at_zero` (`--fp-gradients`) now checks the whole
class automatically: it zeroes each differentiable argument in turn and requires a
finite Jacobian wherever the value is finite.
