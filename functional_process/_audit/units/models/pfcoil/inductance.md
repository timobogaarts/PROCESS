---
kind: model-unit
status: draft
confidence: medium
---

**Ported (partial by switch, full by function).** `pfcoil/inductance.py` /
`test_inductance.py`: `calculate_pf_cs_plasma_inductances` and
`calculate_solenoid_self_inductance` — tier-1. One cottax node, `PFCoilInductance`
(`.tokamak.pf_coil.inductance`), owning `.pf_coil.ind_pf_cs_plasma_mutual`.

**Why it exists**: `.pf_coil.ind_pf_cs_plasma_mutual` was a boundary input of
`models/pfcoil/currents.py::CSFluxSwing` — nothing in the graph wrote it, because
`induct` was unported. It now has a producer.

## source

`process/models/pfcoil.py`:

| lines | what |
|---|---|
| `1721-1984` | `PFCoil.induct`, everything before its `if not output: return` |
| `2837-2867` | `PFCoil.selfinductance`, a `@staticmethod`, pure |

UNPORTED:

| lines | what | why not |
|---|---|---|
| `1767-1778` | the `nohmax` clamp, the `logger.error`, and the `max(noh, 0)` guard for the "FNSF case, noh = -7" TODO | **reproduced since 2026-09-02**, `noh` being computed: `_cs_segments` clips into `[1, NOH_PAD]` — PROCESS's `min(noh, nohmax)`/`max(noh, 0)` with a tighter cap, and a lower bound of `1` rather than `0` because `xohpl / noh` divides by it. The `logger.error` is not ported; the negative-`noh` case remains a PROCESS bug report |
| `1986-2019` | the `output=True` reporting block and the mfile dump | pure reporting, writes nothing to `data` |
| `1944-1947` | `.pf_coil.nef` | `n_cs_pf_coils - 1` on this arm — loop bookkeeping, the same category `pfcoil/__init__.py` records for `n_cs_pf_coils` itself |

## noh is a step function of the CS geometry

`induct` chooses how many pancake segments to split the CS into as

```
noh = ceil(2 * z_pf_coil_upper[CS] / (r_pf_coil_outer[CS] - r_pf_coil_inner[CS]))
```

(`pfcoil.py:1758-1765`), and **every** inductance it returns depends on it: `delzoh` and
`zoh` set where the segments are, `xohpl` and `xohpf` are sums over them divided by
`noh`. So `ind_pf_cs_plasma_mutual` is a piecewise-constant-discontinuous function of the
CS radial thickness — on a run where that thickness is set by an iteration variable.

At the reference point the ratio is `2 x 7.936395 / 0.546817 = 29.027`, i.e. **0.09 %
above the integer 29**, so `noh = 30`. PROCESS's own finite-difference step is
`epsfcn = 1e-3` *relative*, which moves the ratio by 0.1 % — further than the distance
to the step. **PROCESS's own reported derivative of this matrix with respect to the CS
geometry is therefore taken across a discontinuity at this operating point**, and the
solver consumes it. Measured, not argued: perturbing `z_pf_coil_upper[CS]` down by
`1e-3` relative gives a ratio of `28.998` and `noh = 29`.

**This port pinned `noh = 30` and no longer does** (2026-09-02). The pin said a
different `noh` is a different occupant, the way a different `i_pf_location` pattern is.
That is right for a single evaluation and was wrong about the cost. It is right on
`large_tokamak_eval` — the file it was measured on — and wrong on both other solenoid
tokamaks, at their cold *and* their converged designs, by different amounts, so no
single constant works:

| configuration | ratio cold | `noh` cold | ratio converged | `noh` converged |
|---|---|---|---|---|
| `large_tokamak_eval` | 29.028 | **30** | 29.028 | **30** |
| `large_tokamak_nof` | 31.746 | 32 | 26.867 | 27 |
| `low_aspect_ratio_DEMO` | 27.010 | 28 | 26.407 | 27 |

`_cs_segments` now computes it. `noh` stays a **float** — `jnp.ceil` of a traced
quantity is traceable, and the value is only ever divided by, never used as a length —
and the segment arrays are a fixed `NOH_PAD = 64` long with an `active` mask keeping the
arithmetic to exactly `noh` of them, so the trace has one shape for every design.

What it bought, measured cold against PROCESS: `large_tokamak_nof` **669 agreeing / 61
off → 722 / 8**, `low_aspect_ratio_DEMO` **696 / 41 → 723 / 13**, `large_tokamak_eval`
unmoved, `errors` unmoved everywhere, and **no new disagreement on any configuration**.
Eighty of `cold_start`'s eighty-five `NOH_ROWS_*` rows retired; the five that stayed
turned out never to have been `noh` rows at all and are now `PF_COIL_SIX_RESIDUAL`.

**The pinned arm was the one with the wrong derivative** — a correct tangent taken on
the wrong piece. A `ceil` gives the *correct* derivative of a piecewise function, and
confines the error to the value, within one step. Measured before landing it, because a
discontinuity inside an SQP loop deserves more than an argument: the ratio is constant
across every evaluation inside a given SQP window on both files, so **no Jacobian is
ever taken across a step** — the integer changes between iterates, as a pure value
change (`32 → 29 → 27` on `large_tokamak_nof` over seven iterations, `28 → 27` on
`low_aspect_ratio_DEMO`, every crossing while `conv ≥ 4.9e-03`). Driving
`large_tokamak_nof`'s cold start to `7.1e-15` of an integer and perturbing across it
eleven times gives seven iterations and `max|eq| = 2.748e-06` on every row with a
bit-identical objective; the *pinned* arm's objective is not bit-identical across the
same straddle. A ±1e-13 to ±1e-9 jitter table on `.build.dr_cs`, 44 solves, moves no
iteration count, status or residual on either arm.

One real cost: `low_aspect_ratio_DEMO` SAND goes **57 → 79** iterations — to a
*feasible* answer (`min ie −1.85e-07` violated → `+1.18e-11` satisfied) where the pinned
arm returned a marginally infeasible one.

## The cycle, one node larger

`currents.md` records a three-node SCC. Porting `induct` makes it four:

```
PFCoilTimePointCurrents -> PFCoilCurrentWaveform -> PFCoilSizes -> PFCoilInductance
        ^                                                              |
        |                                                   .pf_coil.ind_pf_cs_plasma_mutual
        +------ .pf_coil.f_j_cs_start_end_flat_top ------ CSFluxSwing <-+
```

`PFCoilInductance` reads `.pf_coil.n_pf_coil_turns` and the coil geometry, which
`masses.py::PFCoilSizes` owns; `CSFluxSwing` reads the matrix this node writes. So
`ind_pf_cs_plasma_mutual` stops being an external input and becomes an *internal* edge of
the block.

What that reveals is worth stating plainly: **`pfcoil()`'s `first_call` bootstrap is the
cycle's initial guess.** `pfcoil.py:605-608` sets `ind_pf_cs_plasma_mutual[:, :] = 1.0`
and `n_pf_coil_turns[:] = 100.0` on the first visit and then leans on
`Caller.call_models` re-running the whole pipeline up to ten times. Before this module,
a reader of the port could have taken the matrix for an input someone else supplies.
It is not; it is a state variable of an undeclared fixed-point iteration, and the
`1.0`/`100.0` are its seed. Which `Drive` the four-node block gets is still the
assembler's decision (`currents.md` § "The cycle").

Still **Shape A**: no node reads a `VarPath` it owns, so no `FixedPointFunction` appears
anywhere in this package.

## data footprint

| VarPath | read/write | classification | note |
|---|---|---|---|
| `.physics.rmajor` | read | explicit-arg | `pfcoil.py:1794` — the plasma filament's radius |
| `.physics.ind_plasma` | read | explicit-arg | `:1862`; produced by `models/physics/plasma_inductance.py::PlasmaVoltSecondRequirements` |
| `.build.dr_cs` | read | explicit-arg | `:1814-1815`, for `deltar` only |
| `.build.iohcl` | read | switch | `:1783`, `:1812`, `:1893`, `:1944` |
| `.pf_coil.r_cs_middle` | read | explicit-arg | `:1784`, `:1895` |
| `.pf_coil.r_pf_coil_middle` | read | explicit-arg | `:1870`, `:1919`, `:1954`, `:1956`, `:1977-1978` |
| `.pf_coil.z_pf_coil_middle` | read | explicit-arg | `:1871`, `:1920`, `:1953`, `:1957` |
| `.pf_coil.r_pf_coil_inner` | read | explicit-arg | `:1763`, `:1899` — **index 6 only** |
| `.pf_coil.r_pf_coil_outer` | read | explicit-arg | `:1762`, `:1898` — **index 6 only** |
| `.pf_coil.z_pf_coil_upper` | read | explicit-arg | `:1760`, `:1787`, `:1790`, `:1896`, `:1972` |
| `.pf_coil.z_pf_coil_lower` | read | explicit-arg | `:1972` |
| `.pf_coil.n_pf_coil_turns` | read | explicit-arg | `:1849`, `:1886`, `:1907`, `:1935-1936`, `:1968`, `:1976`, `:1982` |
| `.pf_coil.n_cs_pf_coils`, `.pf_coil.n_pf_cs_plasma_circuits`, `.pf_coil.n_pf_coil_groups`, `.pf_coil.n_pf_coils_in_group` | read | **topology** | loop bounds and array indices |
| `.pf_coil.ind_pf_cs_plasma_mutual` | write | explicit-arg | `:1750` (zeroed), then every entry of the 8x8 circuit block |
| `.pf_coil.nef` | write | **not owned** | `:1944-1947`, loop bookkeeping |

## Whole-array ownership, with the evidence

`.pf_coil.ind_pf_cs_plasma_mutual` is owned **whole**, not per index, and
`_audit/naming_convention.md` § "Array elements" is the reason rather than an exception
to it: per-index addressing is what you reach for when the read range and the write
range of one field *differ*, and here they do not.

- Producer side: `induct` zeroes the entire matrix (`:1750`) and fills every entry of the
  eight-circuit block from one shared set of geometry reads. There is no slice of it this
  node does not compute — unlike `.pf_coil.b_pf_coil_peak`, whose index 6 comes from
  `ohcalc`'s unported CS self-field and which is therefore owned per index in
  `fields.py`.
- Consumer side, in the port: `currents.py::CSFluxSwing` reads `[0:6, 7]`.
- Consumer side, in PROCESS, i.e. what a later pass will need:
  `process/models/pulse.py:228` and `:235` read `[n_cs_pf_coils - 1, ...]`, and
  `process/models/power.py:320-539` reads the whole matrix to build the PF power supply
  circuits. Owning six entries would leave everything those need unowned while this node
  computes it.

## proposed signature(s)

```python
def calculate_solenoid_self_inductance(a, b, c, n) -> float
def calculate_pf_cs_plasma_inductances(
    rmajor, ind_plasma, dr_cs, r_cs_middle,
    r_pf_coil_middle, z_pf_coil_middle, r_pf_coil_inner, r_pf_coil_outer,
    z_pf_coil_upper, z_pf_coil_lower, n_pf_coil_turns) -> jax.Array  # (NGC2, NGC2)
```

`selfinductance` is renamed `calculate_solenoid_self_inductance` per
`naming_convention.md` § "Function/module naming" — the PROCESS name says neither what
it is the self-inductance of nor that it is Bunet's fit, and it is the one name in this
package not already derivable from what it produces.

## cottax node

`PFCoilInductance(ExplicitFunction)`. Occupant for `iohcl = 1` and
`n_pf_coils_in_group = (1, 1, 2, 2)`. **`noh` is not part of the draw** since
2026-09-02 — it is computed inside the occupant, not selected between occupants.

## tier signal

**Tier 1.** No iteration, no CoolProp, no external library; `induct` mutates a
`DataStructure` but only as scratch, and the adapter binds one.

**Sample provenance.** Legacy points read off a converged in-process `SingleRun` of
`large_tokamak_eval.IN.DAT`. The port reproduces PROCESS's full 22x22 matrix
**bit-for-bit** (`relerr = 0.000e+00` on every entry), which is stronger than the
tier-1 tolerance and is what the two reassociations below were chosen to preserve.

**Three arguments are `static_argnames`, and that is a property of PROCESS.**
`z_pf_coil_upper`, `r_pf_coil_inner` and `r_pf_coil_outer` jointly determine `noh`
through a `ceil`. At the reference point the step is 0.09 % away, closer than PROCESS's
own `epsfcn = 1e-3` relative difference step; comparing gradients in those directions
would compare the port's within-piece derivative against a difference quotient taken
*across* the reference's own discontinuity. Not a port defect and not something a
tolerance can absorb. What is lost is one term of Bunet's formula (`c`) and the
`z_cs_half` path: `z_pf_coil_lower` stays differentiated and covers the PF diagonal's
`rl = |z_upper - z_lower|`, and `r_pf_coil_inner`/`r_pf_coil_outer` are read at index 6
only, so no PF coil's geometry goes unchecked.

**Fuzzing** is `±5-20 %` around the reference point with those three held fixed, and
`dr_cs` bounded above `delzoh = 0.5291 m` so the Rosa-Grover split keeps its `sqrt`
branch. Holding them fixed used to be a statement about the *draw* — a draw that changed
`noh` would have been a draw for a different occupant. It is now purely about the
**gradient** comparison: `noh` moving is fine and the value test covers it, but PROCESS's
`epsfcn = 1e-3` difference quotient in those three directions is taken across its own
discontinuity, so there is no reference derivative to compare against. Measured after
the change: 114 gradient cases pass with the three static, the same count as before.

## switches touched

| switch | reachable values | live on `large_tokamak_eval` | decision | evidence |
|---|---|---|---|---|
| `.build.iohcl` | `0`, `1` | `1` | **split** | `:1783` (no CS segments), `:1812-1856` (no CS/plasma coupling), `:1893-1941` (no CS self, no CS/PF), `:1944-1947` (`nef` differs). Four separate blocks disappear |
| `noh` | any positive integer | *computed* | **not a slot** | see § "noh is a step function of the CS geometry" — pinned at `30` until 2026-09-02, now `ceil` of the solved ratio |
| `.pf_coil.n_pf_coils_in_group` pattern | — | `(1, 1, 2, 2)` | **split** (structural) | fixes every array index, as everywhere else in this package |
| `dr_cs >= delzoh` | — | true (`0.5468 >= 0.5291`) | **not** split — a `jnp.where` | `:1814-1819`. This is a numerical guard on a continuous quantity, not a model choice: the same formula with a substituted radicand. Ported as the double-`jnp.where` idiom so the untaken branch's `sqrt` of a negative number cannot leak `nan` into the tangent |

**UNPORTED** for `indat.py`: `iohcl = 0`; any coil-group pattern other than
`(1, 1, 2, 2)`. `noh` was on this list until 2026-09-02 and is not a refusal any more.

## calls into other models

None. `induct` calls `calculate_b_field_at_point` (same file, ported in
`pfcoil/fields.py`) and `PFCoil.selfinductance` (same class, ported here).

## Two exact reassociations

Both were checked against PROCESS bit-for-bit rather than argued from tolerance.

1. **Loop and test roles swapped**, at the CS/plasma and PF/plasma call sites.
   `induct` calls `calculate_b_field_at_point` once per CS segment with a one-element
   loop array (the plasma) and reads `xcin[0]`; the port calls it once with a
   thirty-element loop array and reads all thirty. `ind_mutual_array` depends on the pair
   only through `(r_test + r_loop)**2`, `(z_test - z_loop)**2` and
   `4.0 * r_test * r_loop`, each symmetric under the swap — and the leading `4.0 *` is a
   power of two, so `(4*a)*b` and `(4*b)*a` are the same float. The asymmetric kludge
   (`dr = r_test - r_loop`, `pfcoil.py:5012`) affects only `br`/`bz`, which this unit
   discards. Sixty traced calls become two.
2. **Padded work arrays dropped.** `induct` passes `rc`/`zc`/`cc` of length
   `NGC2 + nohmax = 222`, mostly zeros, and reads only a prefix of the result
   (`xc[0]` at `:1843` and `:1880`, `xc[:noh]` at `:1928`, `xc[:5]` at `:1965-1983`).
   The zero-radius entries produce *non-zero* mutual inductances that are computed and
   discarded. The port builds only the prefix. Checked by reading every index expression
   in the routine, not by assuming the padding is inert.

Both are recorded because they are the kind of change that is obviously safe and
occasionally is not.

## JAX-difficulty flags

- **`math.ceil`** on a traced quantity — the `noh` problem above. Resolved by
  *tracing* it since 2026-09-02: `jnp.ceil` is traceable, the count stays a float
  because it is only ever divided by, and the arrays it would otherwise size are padded
  to `NOH_PAD` and masked. Resolved by making it structural, before that.
- **`if dr_cs >= delzoh`** (`:1814-1819`) → the double `jnp.where`
  (`models/safe_math.py`'s idiom): a single `where` would still evaluate
  `sqrt((dr_cs**2 - delzoh**2) / 12)` on the untaken branch and leak its `nan` into the
  tangent.
- **`3.2 * c * b / a` in Bunet's formula** (`:2866`) — an unguarded division by the
  mean radius. The value survives `a == 0` (`0 / inf` is `0`, and that is also the
  limit) but the derivative comes back `nan`;
  `test_gradient_finite_at_zero` found it. Ported with `safe_math.py`'s double-`where`
  idiom applied to a division rather than a fractional power: bit-identical for every
  `a != 0`, and `0` instead of `nan` at zero, which is the true limit since
  `f ~ k a^3 n^2 / (3.2 c b)` there. The same class of defect as `masses.py`'s
  `sqrt(area)`, found the same way.
- **`math.log(8 * r_pf_coil_middle[k] / rl)`** (`:1978`) — goes to `-inf` as a coil's
  radius goes to zero, and `rl` goes to zero as a coil's height does. Both are physical
  zeros, not boundary points a solver visits, and PROCESS has no guard either. The
  harness's `test_gradient_finite_at_zero` skips them because the *value* is already
  non-finite there, which is the right answer.
- **Summation order** differs: PROCESS accumulates `xohpl`/`xohpf` sequentially,
  `jnp.sum` reduces pairwise. Worth ~1e-16 relative; measured 0 at the reference point.
- No CoolProp, no `scipy.special`, no `scipy.optimize`.

## open questions

- **A structural integer that the solve moves — answered here, still needs a policy.**
  `noh` is not a switch read from the input file; it is `ceil` of a ratio of two solved
  lengths. Pinning it as an occupant was right for a single evaluation and wrong for an
  optimisation that walks across the step, and this unit now **computes it**: pad the
  arrays to a fixed length, mask to the live count, keep the count a float so it stays
  traceable. That is a concrete answer, and it generalises — `cs_fatigue`'s `n_cycle` is
  the same class (`cs_fatigue.md` § marching integration already cites this section) and
  is now masked-`lax.scan`, the same shape. What is still missing is the *policy* saying
  when a structural integer should be padded-and-masked rather than pinned; nothing in
  `naming_convention.md` or `switch_elimination_design.md` covers a structural value
  that changes during a solve. `n_cs_current_filaments` and the coil-group pattern are
  at least fixed by the input file. **Still needs a policy**, now with two worked
  instances rather than none.
- **Should the port smooth `noh`?** Doing so would make the graph differentiable and
  would *not* reproduce PROCESS. The whole point of the harness is that it would then
  fail the value test, loudly, which is the right failure. Flagged only so that nobody
  "fixes" it quietly later.
- **`.pf_coil.nef` is written by PROCESS and by nothing here.** A tier-3/4 comparison
  will see it at its dataclass default. Same category as
  `.pf_coil.p_pf_coil_resistive_total_flat_top` in `masses.md` — either an explicit
  constant output or a `KNOWN_UNVERIFIABLE_OUTPUTS` entry.
- **The last-coil-of-group approximation** (`:1870`, `:1919`): PROCESS evaluates each
  group's plasma and CS mutual inductance at the group's *last* coil and gives every coil
  in the group that same geometric factor. On this run each multi-coil group is an
  up/down symmetric pair, so it is exact; on an asymmetric group it would be an
  unflagged approximation. Reproduced. Worth a note to the physics owners.


## 2026-08-30 (evening) -- the spherical tokamaks' PF coil system, arm 2

`next_steps.md` §18.2 listed five of the eight blockers stopping
`spherical_tokamak_eval.IN.DAT` and `st_regression.IN.DAT` as `pf_coil_system_arm`
deviations (`-1`, `-2`, `-3`, `-6`, `-7`). All five are closed. The package now carries
a `PFCoilTopology` (`models/pfcoil/__init__.py`) instead of five loose module
constants, and `indat._pf_coil_system_arm` has a third positive arm, `2`, for a machine
with **no central solenoid**: `iohcl = 0`, `n_pf_coil_groups = 4`,
`i_pf_location = (2, 3, 3, 4)`, `n_pf_coils_in_group = (2, 2, 2, 2)`,
`i_pf_superconductor = 9`, picture-frame TF. `.tokamak.cs_coil` is `None` on that arm.

**`-3` was a refusal that outlived its cause, and that is a correction to this
record's own frontier.** The predicate refused `itart == 1` *or* `itartpf != 0`.
Measured over `process/`: `itartpf` is read in exactly two places
(`pfcoil.py:1250`, `:411`) and both guard on `itart == 1 **and** itartpf == 0`, and
`core/init.py:640` overwrites `i_pf_location[:3]` under the same conjunction. Both
tracked ST files set `itartpf = 1`, so **neither ever reaches PROCESS's Peng and
Strickler ST arm** -- their PF coil system takes the conventional placement and the
conventional SVD current solve throughout. The predicate is now the conjunction, and
the ST arm stays UNPORTED with nothing reaching it.

**What changed here.** `calculate_pf_plasma_inductances_no_central_solenoid` ports
`induct` at `iohcl = 0`: three of the four blocks survive and one does not. `induct`
guards the CS/plasma block (`pfcoil.py:1812`), the CS self-inductance and the CS/PF
block (`:1893`) on `iohcl != 0`, and sets `nef = n_cs_pf_coils` rather than
`n_cs_pf_coils - 1` (`:1943-1947`) so the PF/PF block covers every coil.

Four reads disappear with those blocks -- `dr_cs` and `r_cs_middle` outright, and
`r_pf_coil_inner`/`r_pf_coil_outer`, whose only use in `induct` is the CS's radial
winding thickness (`:1896-1899`). **`noh` disappears too**, and that closes this
record's own open question *for this arm only*: the piecewise-constant discontinuity in
`dr_cs` that the reference occupant carries is simply not present on a machine with no
solenoid, because `roh`/`zoh` are never filled (`:1783-1791` is guarded) and no
inductance depends on the segment count. The policy gap stands for arms 0 and 1.

`PFCoilInductanceNoCentralSolenoid` is a sibling rather than a subclass, for the same
reason `PFCoilMassesNoCentralSolenoid` is: it declares four reads fewer, and a subclass
may widen a signature but not narrow it. It owns `.pf_coil.ind_pf_cs_plasma_mutual`
whole, on the same producer-side argument. Bit-exact against `induct` in the
scratch verification that produced `TestPFCoilChainSphericalTokamak`'s point; the
matrix itself is not in that contract's return tuple (it is `induct`'s, not
`pfcoil()`'s) and **an ST case in `test_inductance.py` is owed** -- see
`next_steps.md`.

## 2026-08-31 -- the no-central-solenoid arm has a harness case

`calculate_pf_plasma_inductances_no_central_solenoid` was verified bit-exact in a
scratch script when it landed (2026-08-30) and nothing in the test tree held it;
`tests/functional_process/models/pfcoil/test_inductance.py::
TestCalculatePfPlasmaInductancesNoCentralSolenoid` does now, against `PFCoil.induct(False)`
at `iohcl = 0` on the eight-coil topology. It agrees at the legacy point and under fuzz,
and passes every gradient check (`--fp-gradients`, 30 tests over this file, 76 s).

**`noh` is computed before `induct` looks at `iohcl`, and on arm 2 it divides by the last
PF coil.** `pfcoil.py:1756-1780` reads
`2 * z_pf_coil_upper[n_cs_pf_coils - 1] / (r_pf_coil_outer[...] - r_pf_coil_inner[...])`
unconditionally. With no solenoid, index `n_cs_pf_coils - 1` is index 7 -- the last PF
coil, not the CS. The quotient is then never used (`roh`/`zoh` are filled only under
`iohcl != 0` and all three blocks reading them are guarded), so no inductance depends on
it, which is what §20.6 already recorded. Two live consequences the harness adapter had
to face, both stated in its docstring rather than worked around:

- the two radii must be seeded to something with a non-zero difference, or PROCESS
  raises `ZeroDivisionError` computing a number it will not use;
- on the spherical tokamaks' geometry `z_pf_coil_upper[7]` is **negative** (coil 7 is
  below the midplane), so `math.ceil` gives a negative `noh` and `max(noh, 0)`
  (`:1778` -- PROCESS's own FNSF guard, carrying its own `TODO` that "noh should always
  be positive") clamps it to zero. That guard, written for a different machine, is what
  keeps `roh = np.zeros(noh)` legal here.

The contract therefore declares **no `static_argnames`**, unlike its `iohcl = 1` sibling:
the three that contract holds static are the three `noh` steps on, and `noh` reaches no
output on this arm, so `z_pf_coil_upper` is an ordinary differentiated input whose only
use is the PF/PF diagonal's `rl = |z_upper - z_lower| / sqrt(pi)`.

## 2026-08-31 -- every pair at once

`induct` evaluates the same Green's-function kernel at many points, and the port traced
it once per point: four times for the CS/PF-group inductances (thirty CS filaments seen
from each group's representative coil), six times for the PF/PF block (each coil against
the other five), twice for the inner/outer CS filament radii against the plasma. Thirteen
traces of a ~110-equation kernel, 1,960 equations in this module plus its share of
`fields.py`'s.

Three batched forms replace them, none of which changes an expression inside
`_mutual_inductances`:

- `_mutual_inductances_over_test_points`, `in_axes=(None, None, 0, 0)` -- the CS/PF-group
  block, four test points over one thirty-filament set, and the PF/PF block.
- `_mutual_inductances_over_loop_sets`, `in_axes=(0, None, None, None)` -- the inner and
  outer CS radii against the one plasma filament.
- The PF/PF block is now the **full `(n_pf_coils, n_pf_coils)` matrix in one call**,
  self pairs included. A coil against itself is finite -- `_S_MAX` clamps `s` at
  `0.999999` before `log(1/(1 - s))`, and its tangent is finite for the same reason -- and
  the diagonal is overwritten by the thin-ring self-inductance immediately afterwards, so
  computing it costs one column of arithmetic and saves five traces of the kernel.

The `2 * n_pf_coils * (n_pf_coils - 1)` scattered `.at[i, k].set` writes become one
`.at[:n, :n].set` of a block; the plasma and CS columns become one paired write each.

### Reductions reassociated by vectorisation

`xohpf` was `jnp.sum` over a thirty-element vector per group and is now `jnp.sum(..., axis=1)`
over a `(4, 30)` block, which XLA is free to reduce in a different order. Expect the CS/PF
mutual inductances to differ from the unrolled version in the last bits. Everything else
here is elementwise and unchanged; the tier-1 contracts pass at their own tolerances.

This module's share of `large_tokamak_nof` fell from 1,960 equations to 418, and it
accounts for most of `fields.py`'s remaining drop as well.
