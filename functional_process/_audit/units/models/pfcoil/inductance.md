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
| `1767-1778` | the `nohmax` clamp, the `logger.error`, and the `max(noh, 0)` guard for the "FNSF case, noh = -7" TODO | with `noh` a graph-assembly constant there is nothing left to clamp; the negative-`noh` case is a different occupant and a PROCESS bug report |
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

This port fixes `noh = 30` as a module constant, i.e. a different `noh` is a different
occupant, the same way a different `i_pf_location` pattern is. Three arguments are
consequently `static_argnames` in the contract — see § tier signal. The wider problem is
in § open questions: the conventions cover a switch read from input, and this is a
structural integer that the solve itself moves.

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

`PFCoilInductance(ExplicitFunction)`. Occupant for `iohcl = 1`,
`n_pf_coils_in_group = (1, 1, 2, 2)` and `noh = 30`.

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

**Fuzzing** is `±5-20 %` around the reference point with those three held fixed (a draw
that changed `noh` would be a draw for a different occupant), and `dr_cs` bounded above
`delzoh = 0.5291 m` so the Rosa-Grover split keeps its `sqrt` branch.

## switches touched

| switch | reachable values | live on `large_tokamak_eval` | decision | evidence |
|---|---|---|---|---|
| `.build.iohcl` | `0`, `1` | `1` | **split** | `:1783` (no CS segments), `:1812-1856` (no CS/plasma coupling), `:1893-1941` (no CS self, no CS/PF), `:1944-1947` (`nef` differs). Four separate blocks disappear |
| `noh` | any positive integer | `30` | **split** (structural) | see § "noh is a step function of the CS geometry" |
| `.pf_coil.n_pf_coils_in_group` pattern | — | `(1, 1, 2, 2)` | **split** (structural) | fixes every array index, as everywhere else in this package |
| `dr_cs >= delzoh` | — | true (`0.5468 >= 0.5291`) | **not** split — a `jnp.where` | `:1814-1819`. This is a numerical guard on a continuous quantity, not a model choice: the same formula with a substituted radicand. Ported as the double-`jnp.where` idiom so the untaken branch's `sqrt` of a negative number cannot leak `nan` into the tangent |

**UNPORTED** for `indat.py`: `iohcl = 0`; any `noh != 30`; any coil-group pattern other
than `(1, 1, 2, 2)`.

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

- **`math.ceil`** on a traced quantity — the `noh` problem above. Resolved by making it
  structural, not by tracing it.
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

- **A structural integer that the solve moves.** `noh` is not a switch read from the
  input file — it is `ceil` of a ratio of two solved lengths. The port's answer ("a
  different `noh` is a different occupant") is right for a single evaluation and wrong
  for an optimisation that walks across the step. Nothing in
  `naming_convention.md` or `switch_elimination_design.md` covers a structural value
  that changes *during* a solve. This is the sharpest instance of that problem the port
  has hit; `n_cs_current_filaments` and the coil-group pattern are at least fixed by the
  input file. **Needs a policy**, not a per-unit decision.
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
