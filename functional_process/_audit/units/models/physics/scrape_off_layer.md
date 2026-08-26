---
kind: model-unit
status: draft
confidence: medium
---

**Ported (2026-08-26).** `scrape_off_layer.py` / `test_scrape_off_layer.py`: all five
entered functions (`tokamak_call_surface.md` §A row 2.7: "5 entered functions, 226
entered LOC, unported" -- `models/tokamak/namespace.py:157-159`, the row this record
resolves).

**Not on the traced boundary today.** `_audit/tokamak_boundary.md`'s "The 58 that are the
work list" table attributes **zero** boundary reads to `.tokamak.scrape_off_layer` --
recorded there as a real result, not a gap. Confirmed independently while writing this
record: see "who reads this, and who does not" below. Ported anyway, per the wave-1
brief's scope (trace the chain and report who reads it, rather than skip a slot the
boundary tool currently shows as unconsumed) and because it is cheap, self-contained, and
a future consumer (or `output()` reporting parity) may need it.

## source

`process/models/physics/scrape_off_layer.py`, 329 lines, full file in scope except
`output()` (100-171, pure reporting, no `data` writes beyond the fields already produced
by `run()`).

| # | function | lines | shape |
|---|---|---|---|
| 1 | `ScrapeOffLayer.run` | 22-98 | the stateful shell; one computes-then-selects switch |
| 2 | `ScrapeOffLayer.output` | 100-171 | reporting shell, out of scope |
| 3 | `ScrapeOffLayer.calculate_eich2013_sol_power_decay_length` | 173-216 | `@staticmethod`, pure |
| 4 | `ScrapeOffLayer.calculate_mast2014_sol_power_decay_length_1` | 218-254 | `@staticmethod`, pure |
| 5 | `ScrapeOffLayer.calculate_mast2014_sol_power_decay_length_2` | 256-284 | `@staticmethod`, pure |
| 6 | `ScrapeOffLayer.calculate_upstream_sol_outboard_parallel_area` | 286-328 | `@staticmethod`, pure, called twice |

Functions 3-6 are already `@staticmethod`s taking plain floats -- `CallableNode.fn`
already, needing only the `safe_pow` treatment (see JAX-difficulty flags). The seam is
therefore exactly at `run()`'s boundary, and `run()` itself is short and entirely
unconditional except for one three-way selection (lines 46-72).

## the RAW mint caution

**`ScrapeOffLayer.run()` is called from `Physics.run()` at `process/models/physics/
physics.py:832`, which reads `self.data.physics.p_plasma_separatrix_mw` at that point --
before the positivity kludge at `physics.py:839-845` runs.** Verified by reading
`physics.py:790-845` directly:

- `:800` -- `self.data.physics.p_plasma_separatrix_mw = self.exhaust.calculate_
  separatrix_power(...)` -- the raw write.
- `:832` -- `self.scrape_off_layer.run()` -- this unit's entry point, reading
  `self.data.physics.p_plasma_separatrix_mw` at lines 24-42 of `scrape_off_layer.py`,
  three times (once per length calculation), plus twice more in the two flux divisions
  (lines 90-98).
- `:839-845` -- `# KLUDGE: Ensure p_plasma_separatrix_mw is continuously positive` --
  `self.data.physics.p_plasma_separatrix_mw /= 1 - np.exp(-self.data.physics.p_plasma_
  separatrix_mw)` -- the transform, *after* `scrape_off_layer.run()` has already run.

So every read of the separatrix power inside this unit is of the **pre-kludge (raw)**
value. `functional_process/models/physics/physics.py` (another agent's file this wave)
mints this distinction explicitly: `SeparatrixPowerNonIgnited` owns
`.physics.p_plasma_separatrix_mw_raw`, and `PositiveSeparatrixPower` (one node later)
owns the real, transformed `.physics.p_plasma_separatrix_mw` that every consumer *after*
`physics.py:845` sees. Every node in this unit that reads the separatrix power declares
`p_plasma_separatrix_mw_raw=From(physics)`, never `p_plasma_separatrix_mw` -- pinned by
`tests/functional_process/models/physics/test_scrape_off_layer.py::
test_raw_separatrix_power_nodes_read_the_raw_mint`.

## who reads this, and who does not

Grep for every `VarPath` this unit writes (`len_plasma_sol_eich13_power_decay`,
`len_plasma_sol_mast14_power_decay_1`, `len_plasma_sol_mast14_power_decay_2`,
`len_sol_outboard_power_decay`, `a_plasma_outboard_sol_parallel`,
`a_plasma_outboard_sol_eich13_parallel`, `pflux_plasma_outboard_sol_parallel_mw`,
`pflux_plasma_outboard_sol_eich13_parallel_mw`) across `process/` outside this file and
its own `output()`:

```
process/models/physics/physics.py:1033:  self.data.physics.len_sol_outboard_power_decay
process/core/io/plot/summary.py:9023-9037:  mfile.get("len_plasma_sol_{eich13,mast14_*}_power_decay")
```

**One real reader, `physics.py:1033`.** Inside `Physics.run()`'s divertor-imbalance block
(unconditional, no switch guard on the read itself): `fio = 0.16 + (0.16-0.41) * (1 -
2/(1+exp(-(drsep/len_sol_outboard_power_decay)**2)))`. `fio` in turn feeds `fli`, `flo`,
`plimw`, `plomw` (`physics.py:1041-1068`), and **every one of those five fields is read
only by `Physics.output()`'s reporting** (`physics.py:2230-2253`, `po.ovarre` calls) --
grep confirms no constraint (`process/core/solver/constraints.py`) and no other model's
`run()` reads `fio`/`fli`/`flo`/`plimw`/`plomw`. So the one real consumer of this unit's
output is itself purely a reporting computation, not a solve-path dependency -- consistent
with `tokamak_boundary.md`'s "zero boundary reads" finding, since that tool traces the
*compute* graph, not `output()`.

**No constraint reads anything this unit writes.** Grepped `process/core/solver/
constraints.py` for all eight `VarPath`s above (plus `i_len_sol_outboard_power_decay`):
zero matches.

**Not ported here**: the `fio`/`fli`/`flo`/`plimw`/`plomw` chain itself -- it belongs to
`physics.py`'s own scope (another agent's file this wave), not this unit's, and is purely
reporting-consumed so is not blocking anything on the traced graph either way.

## data footprint

Reference run: `tests/regression/input_files/large_tokamak_eval.IN.DAT` --
`i_len_sol_outboard_power_decay` unset, so PROCESS's own default `1` (`EICH_2013`,
`physics_variables.py:1718`) applies.

| VarPath | read/write | classification | note |
|---|---|---|---|
| `.physics.p_plasma_separatrix_mw` (RAW, see above) | read | explicit-arg | `:24,31,38,90,95` -- ported as `.physics.p_plasma_separatrix_mw_raw` |
| `.physics.rmajor` | read | explicit-arg | `:26,75` |
| `.physics.b_plasma_surface_poloidal_average` | read | explicit-arg | `:27,32,78,79` |
| `.physics.aspect` | read | explicit-arg | `:28` |
| `.physics.plasma_current` | read | explicit-arg | `:40`, converted A -> MA at the call site |
| `.physics.rminor` | read | explicit-arg | `:75` |
| `.physics.b_plasma_outboard_total` | read | explicit-arg | `:78` |
| `.physics.i_len_sol_outboard_power_decay` | read | switch | `:46-72`, *(live, `=1`, PROCESS default)* |
| `.physics.len_plasma_sol_eich13_power_decay` | **write** | explicit-arg | `:24-29`, unconditional |
| `.physics.len_plasma_sol_mast14_power_decay_1` | **write** | explicit-arg | `:31-34`, unconditional |
| `.physics.len_plasma_sol_mast14_power_decay_2` | **write** | explicit-arg | `:37-42`, unconditional |
| `.physics.len_sol_outboard_power_decay` | **write** | switch (select) | `:46-72`, one of the three above, or left untouched (`USER_INPUT`, `==0`) |
| `.physics.a_plasma_outboard_sol_parallel` | **write** | explicit-arg | `:74-80`, unconditional, reads the *selected* length |
| `.physics.a_plasma_outboard_sol_eich13_parallel` | **write** | explicit-arg | `:82-88`, unconditional, reads the Eich length *directly*, regardless of the switch |
| `.physics.pflux_plasma_outboard_sol_parallel_mw` | **write** | explicit-arg | `:90-93`, unconditional |
| `.physics.pflux_plasma_outboard_sol_eich13_parallel_mw` | **write** | explicit-arg | `:95-98`, unconditional |

## proposed / actual signature(s)

```python
def calculate_eich2013_sol_power_decay_length(p_plasma_separatrix_mw, rmajor, b_plasma_surface_poloidal_average, aspect) -> float
def calculate_mast2014_sol_power_decay_length_1(p_plasma_separatrix_mw, b_plasma_surface_poloidal_average) -> float
def calculate_mast2014_sol_power_decay_length_2(p_plasma_separatrix_mw, cur_plasma_ma) -> float
def calculate_upstream_sol_outboard_parallel_area(rmajor, rminor, len_plasma_sol_power_decay, b_plasma_outboard_total, b_plasma_surface_poloidal_average) -> float
def calculate_scrape_off_layer(p_plasma_separatrix_mw_raw, rmajor, rminor, b_plasma_surface_poloidal_average, b_plasma_outboard_total, aspect, plasma_current, i_len_sol_outboard_power_decay) -> tuple[float, ...]  # composite, all 8 outputs; test-file boundary only, see § cottax nodes
```

No signature changes from PROCESS's own `@staticmethod`s for functions 3-6; `calculate_
scrape_off_layer` is new, composing them (mirrors `confinement_time.py`'s `calculate_
confinement_time`).

## cottax nodes

`functional_process/models/physics/scrape_off_layer.py`:

| class | family | owns | reads |
|---|---|---|---|
| `Eich2013SOLPowerDecayLength` | (none -- unconditional) | `.physics.len_plasma_sol_eich13_power_decay` | `.physics.p_plasma_separatrix_mw_raw`, `.physics.rmajor`, `.physics.b_plasma_surface_poloidal_average`, `.physics.aspect` |
| `Mast2014SOLPowerDecayLength1` | (none -- unconditional) | `.physics.len_plasma_sol_mast14_power_decay_1` | `.physics.p_plasma_separatrix_mw_raw`, `.physics.b_plasma_surface_poloidal_average` |
| `Mast2014SOLPowerDecayLength2` | (none -- unconditional) | `.physics.len_plasma_sol_mast14_power_decay_2` | `.physics.p_plasma_separatrix_mw_raw`, `.physics.plasma_current` |
| `OutboardSOLPowerDecayLength` | family base | -- | -- |
| `OutboardSOLPowerDecayLengthEich2013` | `OutboardSOLPowerDecayLength` (`EICH_2013`, live) | `.physics.len_sol_outboard_power_decay` | `.physics.len_plasma_sol_eich13_power_decay` (only) |
| `UpstreamSOLOutboardParallelArea` | (none -- unconditional) | `.physics.a_plasma_outboard_sol_parallel` | `.physics.rmajor`, `.physics.rminor`, `.physics.len_sol_outboard_power_decay`, `.physics.b_plasma_outboard_total`, `.physics.b_plasma_surface_poloidal_average` |
| `UpstreamSOLOutboardEich13ParallelArea` | (none -- unconditional) | `.physics.a_plasma_outboard_sol_eich13_parallel` | `.physics.rmajor`, `.physics.rminor`, `.physics.len_plasma_sol_eich13_power_decay`, `.physics.b_plasma_outboard_total`, `.physics.b_plasma_surface_poloidal_average` |
| `OutboardSOLParallelPowerFlux` | (none -- unconditional) | `.physics.pflux_plasma_outboard_sol_parallel_mw` | `.physics.p_plasma_separatrix_mw_raw`, `.physics.a_plasma_outboard_sol_parallel` |
| `OutboardSOLEich13ParallelPowerFlux` | (none -- unconditional) | `.physics.pflux_plasma_outboard_sol_eich13_parallel_mw` | `.physics.p_plasma_separatrix_mw_raw`, `.physics.a_plasma_outboard_sol_eich13_parallel` |

Registration: all eight classes are proposed as sub-nodes of `.tokamak.scrape_off_layer`
(`models/tokamak/namespace.py:157`) -- see the final report for the exact instructions
this record does not itself carry out (wiring is the consolidation pass's job).

## tier signal

**Tier 1, all functions and all node classes.** No `scipy.optimize`, no `fsolve`, no ad
hoc fixed-iteration loop, no CoolProp. `calculate_upstream_sol_outboard_parallel_area` is
called into by another model only in the trivial sense of being called twice by this same
unit's own `run()`; nothing in the file touches another `Model` instance.

**Sample provenance.** Functions 3-6 have exact-value legacy samples from `tests/unit/
models/physics/test_scrape_off_layer.py` (parametrised cases plus one `_exact` case
each), reused verbatim. `calculate_scrape_off_layer` (the composite) has no PROCESS
bare-function counterpart to lift a legacy sample from, so its "legacy" points are
self-authored, physically-plausible operating points in the spirit of `confinement_
time.md`'s `_BASE` (documented there as "not a converged PROCESS run... but plausible
... magnitudes"), diffed against a real `ScrapeOffLayer.run()` via a `DataStructure`
adapter rather than against a hardcoded expected value.

## switches touched

One, in `run()`.

| switch | reachable values | live on `large_tokamak_eval` | decision | evidence |
|---|---|---|---|---|
| `.physics.i_len_sol_outboard_power_decay` | `0` (`USER_INPUT`), `1` (`EICH_2013`), `2` (`MAST_2014_1`), `3` (`MAST_2014_2`) | `1` (unset, PROCESS default `physics_variables.py:1718`) | **split**, computes-then-selects | All three of `EICH_2013`/`MAST_2014_1`/`MAST_2014_2`'s underlying lengths are computed by PROCESS **unconditionally**, regardless of this switch's value (`run()`'s three length calls precede the `if`/`elif`/`elif`, lines 24-42 vs 46-72) -- the switch only decides which already-computed value is passed through to `len_sol_outboard_power_decay`. Per the wave-1 settled policy ("computes-then-selects families -> one occupant class per switch value, each declaring only its own arm's reads"), each value is still its own occupant even though every occupant's body is a one-line passthrough: a union node would read all three candidates unconditionally, which is exactly the invented-edge shape the split exists to remove, even though here the *values themselves* (not just the reads) are already all computed regardless -- the switch's only real effect is which single `VarPath` feeds `len_sol_outboard_power_decay` downstream. `EICH_2013` is ported (`OutboardSOLPowerDecayLengthEich2013`, live); `MAST_2014_1`/`MAST_2014_2` are UNPORTED (same one-line shape, not needed by the reference arm); `USER_INPUT` (0) is not a computation in PROCESS at all -- there is no `else` arm, so that value leaves the field at whatever it already was (the input file's own value, or the `DataStructure` default) and has no occupant to write |

`calculate_scrape_off_layer` (the composite test-file boundary function) reproduces this
exactly for the three computed values and raises `ValueError` for `USER_INPUT` -- a
**declared divergence** from PROCESS (which does not raise there), documented in the
function's own docstring and pinned by `test_user_input_switch_value_is_not_ported`,
since the composite has no PROCESS answer to reproduce for an arm that does not compute
anything.

## calls into other models

None. `run()` calls only `self.<staticmethod>`, called twice for `calculate_upstream_
sol_outboard_parallel_area`. No reverse calls either (unlike `plasma_geometry.md`'s
`calculate_iter_physics_basis_elongation`, nothing outside this file calls into
`scrape_off_layer.py`).

## JAX-difficulty flags

- **Fractional powers, all wrapped in `safe_pow`**: `p_plasma_separatrix_mw**-0.02`,
  `rmajor**0.04`, `b_plasma_surface_poloidal_average**-0.92`, `aspect**-0.42` (Eich 2013);
  `p_plasma_separatrix_mw**0.18`, `b_plasma_surface_poloidal_average**-0.68` (MAST 1);
  `p_plasma_separatrix_mw**0.22`, `cur_plasma_ma**-0.64` (MAST 2). **`rmajor**0.04` is
  wrapped here even though `divertor.md`'s structurally-identical `lambda_eich` term left
  its own `rmajor**0.04` bare** -- that exemption was incidental there (zeroing `rmajor`
  already made `divwade`'s *value* non-finite via an unrelated `0/0` in
  `atan(bp_omp/bt_omp)`, so `test_gradient_finite_at_zero`'s own "skip if the value is
  already non-finite" branch applied before the power law was ever reached), and
  `calculate_eich2013_sol_power_decay_length` has no such coincidental second
  singularity -- zeroing `rmajor` here leaves every other factor finite, so an unwrapped
  `rmajor**0.04` would genuinely fail the check. Confirmed by running `--fp-gradients`
  with the wrapper in place (see § deviations); not re-confirmed by deliberately removing
  it, per the brief's "no static switch kwargs may be guessed" spirit applied to this
  smaller claim too -- the reasoning is measured on `divwade`'s case and inferred, not
  independently reproduced by breaking this file.
- **`calculate_upstream_sol_outboard_parallel_area`'s division by `b_plasma_outboard_
  total`**: unguarded, but *not* registered in `_harness/boundary.py`, because zeroing
  `b_plasma_outboard_total` makes the function's **value** itself non-finite (`b/0 ->
  inf`), not just its gradient -- that is `test_outputs_finite`'s domain, and `test_
  gradient_finite_at_zero`'s own "skip if value is non-finite" branch owns it structurally
  before any registration would be needed. Confirmed by running the harness with `--fp-
  gradients`: no failure, no registration required (see § deviations for the actual run).
- No in-place mutation, no dynamic shapes, no traced-value Python `if` (the one branch,
  `i_len_sol_outboard_power_decay`, is a switch, resolved at graph-assembly time, not a
  traced quantity), no `ProcessValueError`/`ValueError` on a traced condition in the
  ported scope (`run()`'s missing `else` for `USER_INPUT` is a no-op, not a raise, in
  PROCESS itself).

## open questions

1. **Should the `fio`/`fli`/`flo`/`plimw`/`plomw` reporting chain (`physics.py:1023-
   1068`) ever be ported**, given it is this unit's only real consumer and is itself
   consumed only by reporting? Not decided here -- it is `physics.py`'s own scope this
   wave (another agent's file), and this record only notes that the chain exists and
   where it lives.
2. **Registration shape**, same open question `divertor.md` left: whether the eight
   classes above should be flat sub-nodes of `.tokamak.scrape_off_layer` or grouped
   under intermediate namespaces mirroring the family structure -- left to the
   consolidation pass, see the final report.

## deviations from PROCESS

- `calculate_scrape_off_layer` raises `ValueError` for `i_len_sol_outboard_power_decay
  == USER_INPUT` (0), where PROCESS silently leaves the field untouched. See § switches
  touched. This affects only the composite test-file boundary function, not the node
  classes (which have no occupant for that value at all -- there is nothing to call).
- No other deviations. Every ported pure function is bit-identical to PROCESS's own
  `@staticmethod` except for the `safe_pow` wrapping, which is value-identical by
  construction (`safe_math.py`'s own docstring).

## test results

`$PY -m pytest tests/functional_process/models/physics/test_scrape_off_layer.py`:
**53 passed, 49 skipped** on a plain run (gradient checks skip by default). **102
passed** with `--fp-gradients` -- including `test_gradient_finite_at_zero` for every
contract, so the `safe_pow` wrapping described above (in particular `rmajor**0.04`,
wrapped where `divertor.md`'s analogous term was not) is measured sufficient: zero
`_harness/boundary.py` registrations were needed for this unit.
