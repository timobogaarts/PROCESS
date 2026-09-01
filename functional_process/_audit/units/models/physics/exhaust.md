---
kind: model-unit
status: draft
confidence: high
---

**Ported.** `exhaust.py` now declares `calculate_radiation_fraction`, unchanged from
source, plus one cottax node (`RadiationFraction`).

## source

`process/models/physics/exhaust.py` (221 lines). Registry unit #11, in-scope method
`calculate_radiation_fraction` (L194-220). The file's other three `@staticmethod`s
(`calculate_separatrix_power`, `calculate_psep_over_r_metric`,
`calculate_eu_demo_re_attachment_metric`, L88-192) are already pure, self-contained, no
`self.data` access — but not in the registry's stated scope for this unit and not called
by `calculate_radiation_fraction` itself, so not ported here (a mechanical follow-up if
this unit's scope is ever widened, not a blocker on anything). `PlasmaExhaust.run()` is
a no-op; `PlasmaExhaust.output()` is a pure reporting shell, out of scope by the schema's
own convention (formatted-output writers are never pure-port candidates).

## data footprint

`calculate_radiation_fraction` is already a clean `@staticmethod` in source — no
`self.data` access at all, both arguments explicit.

| VarPath | read/write | classification | note |
|---|---|---|---|
| `.physics.p_plasma_rad_mw` | read | explicit-arg | |
| `.physics.p_plasma_heating_total_mw` | read | explicit-arg | bound to the port's `p_plasma_heating_mw` parameter — a call-site rename (source's own parameter is spelled `p_plasma_heating_mw`, matching neither PROCESS field exactly; kept as source spells it, same move `confinement_time.md` made for `zeff`/`ntau`) |
| `.physics.f_p_plasma_separatrix_rad` | write (by caller) | — | both call sites (`stellarator.py:2369-2373`, `physics.py:1080-1085`, unit #9) assign the return value directly onto this field; `calculate_radiation_fraction` itself performs no write |

## proposed signature(s)

```python
def calculate_radiation_fraction(
    p_plasma_rad_mw: float, p_plasma_heating_mw: float
) -> float: ...
```
Unchanged from source's own signature and body shape (one domain guard, one division).

## cottax node

```python
from cottax.interfaces.pytree_namespace_module import ExplicitFunction, From, OutputInto
from functional_process.paths import physics


class RadiationFraction(ExplicitFunction):
    f_p_plasma_separatrix_rad = OutputInto(physics)

    def __call__(
        self,
        p_plasma_rad_mw=From(physics),
        p_plasma_heating_total_mw=From(physics),
    ):
        return calculate_radiation_fraction(p_plasma_rad_mw, p_plasma_heating_total_mw)
```
Registered in `exhaust.py`, not yet wired into `total_process.py` — reserved for the
consolidation pass per this wave's boundary.

## tier signal

**Tier 1.** No internal iteration, no call into any other model, one domain guard
(`p_plasma_heating_mw == 0`) handled as an ordinary `jnp.where` rather than a raise (see
"JAX-difficulty flags").

## switches touched

None. `calculate_radiation_fraction` reads no `data.<area>.i_*` field.

## calls into other models

None directly. **Caller-side dependency, not a call from this unit**: both real call
sites (`stellarator.py:2369-2373`, `physics.py:1080-1085`) feed this function's
`p_plasma_heating_mw` argument from `.physics.p_plasma_heating_total_mw`, itself the
output of `physics.py`'s `calculate_total_plasma_heating_power` — one of unit #9's
in-scope methods, currently being audited by another agent in this same wave. This is a
data dependency at the *caller's* call site, not a call this unit's own body makes, so
it does not block porting `calculate_radiation_fraction` itself (its own signature takes
the value as a plain argument, same as every other unit's "close the `data` back door"
treatment) — flagged for whoever wires this node's read up against unit #9's eventual
`total_process.py` registration, not resolved here.

## JAX-difficulty flags

- **`p_plasma_heating_mw == 0` domain guard** — `workaround-known`. Source returns a
  real, finite `0.0` (plus a logged warning) rather than raising, so this is *not* the
  `reference_domain_errors` case (`test_harness.md`'s tier-1 domain-guard convention is
  for a raise PROCESS signals invalid input with — this is PROCESS choosing a defined
  fallback value instead). Ported as `jnp.where` with a **safe denominator**
  (`jnp.where(zero_heating, 1.0, p_plasma_heating_mw)`) rather than a bare
  `p_plasma_rad_mw / p_plasma_heating_mw` inside the outer `where` — the latter would
  compute `x / 0.0` on the untaken branch, which back-propagates a NaN gradient into the
  *taken* branch as well (the exact `jnp.where`-leaks-NaN failure mode
  `test_harness.md`'s worked example and `next_steps.md`'s pinned regression test both
  exist to catch). Verified directly: `test_gradient_finite`/`test_gradient_agreement`
  pass at the `zero-heating-power` sample with this construction.
- No CoolProp calls, no `scipy.optimize`/`fsolve`, no `copy.deepcopy`, no switches.

## open questions

None outstanding for this unit.

## 2026-08-27 — `calculate_eu_demo_re_attachment_metric` ported (missing-producer wave)

`optimise_design.md` §11.5's constraint-68 row, and the sharpest single instance of
that whole defect class. `.physics.p_div_bt_q_aspect_rmajor_mw` was a boundary constant
at `0` against PROCESS's converged `10.4949`, so:

- §11.3's Stage A value nevertheless agreed **bit for bit** with PROCESS
  (`+4.949055142e-02` normalised residual, identical), because the warm seed handed the
  port PROCESS's own converged `DataStructure`;
- §11.4's Stage B gradient row read `1.00e+00` relative — the port's total derivative
  was **identically zero** where PROCESS's was not;
- §11.6's Stage C2 then failed outright: pyvmcon's first QP raised
  `QSPSolverException`, because c68 is *violated* by +4.9% at PROCESS's own answer and a
  violated constraint with an identically zero linearised row admits no feasible step.

A value test could not have found it and a gradient test could. That is the case for
the harness, made by a live measurement rather than by an injected `stop_gradient`.

The function is `exhaust.py:150-192` unchanged. The node reads the **mint**
`.physics.p_plasma_separatrix_mw_raw`, not the field: `physics.py:818-826` runs before
the KLUDGE at `:843-845`, and `physics.md`'s `force_positive_separatrix_power` entry
already names this as one of the three call sites that see the pre-transform number. At
this machine `P_sep = 176.8 MW` makes the two readings differ by ~1e-77 and no test
could tell them apart — which is exactly why the wiring was done from the source rather
than from the agreement.

This narrows, and does not withdraw, the module's original "out of the registry's
stated scope" note. `calculate_separatrix_power` was already ported (in `physics.py`,
beside the mint it feeds). `calculate_psep_over_r_metric` stays unported on purpose: no
active constraint and no ported node reads `.physics.p_plasma_separatrix_rmajor_mw`, so
an occupant would be a producer with no consumer — a two-line follow-up the day one
appears, the same disposition `PlasmaEnergyFromBeta` records for
`.physics.e_plasma_beta_thermal`.

Registered as a sixth slot of `.tokamak.physics` (`re_attachment_metric`), unswitched,
so an instance default. Tier 1; `test_exhaust.py::TestEuDemoReAttachmentMetric` diffs
the real staticmethod at the violating operating point, green plain and under
`--fp-gradients`. No cycle created.

## 2026-09-01 — `calculate_psep_over_r_metric` ported (missing-producer wave 2)

**The section above ends with a claim that was already false when it was written, and
this section is its correction.** It says `calculate_psep_over_r_metric` "stays unported
on purpose: no active constraint and no ported node reads
`.physics.p_plasma_separatrix_rmajor_mw`, so an occupant would be a producer with no
consumer." Constraint 56 is active on `spherical_tokamak_eval.IN.DAT` (`:21`) and
`st_regression.IN.DAT` (`:689`) and reads exactly that path. Both files were already in
`run_cold_matrix.CONFIGURATIONS` on 2026-08-27.

**Why the claim survived four days of pins that were all looking at this.** It is
`optimise_design.md` §26.1's finding and it is structural, not an oversight:
`reference_boundary*.txt`, `missing_producers_tokamak.txt`,
`reference_provider_*.txt` and `boundary.unproduced_but_computed` are every one of them
measured on `driven_graph(graph_for(...))` — the **models**. The objective and the
constraint nodes are inserted later, by `mdf.mdf_graph`/`sand.optimise_graph`. A path
read *only by a condition* is invisible to all four pins, and this is one:
`reference_provider_st_regression.txt` reported **one** `computed` row where the same
measurement over `mdf_graph`'s graph reports four. The lesson generalises past this
unit — a "nothing reads it" note in this port means "no *model* reads it" unless it
says otherwise.

The function is `exhaust.py:127-147` unchanged, one division. The node
(`PsepOverRMetric`) reads the **mint** `.physics.p_plasma_separatrix_mw_raw` for the
identical reason its neighbour does, and the reason is slightly stronger here:
`physics.py:811-816` is the *first* of the three call sites that see
`.physics.p_plasma_separatrix_mw` before the KLUDGE at `:843-845`, sitting three lines
above the re-attachment metric in the same block. At `P_sep` = 180.0/181.3 MW the two
readings differ by ~1e-79 and no test in this port can tell them apart; the wiring is
from the source, not from the agreement.

**What the freeze cost, per file, measured against PROCESS's converged
`DataStructure`:**

| file | port before | PROCESS | c56 bound | verdict before | verdict at PROCESS's answer |
|---|---|---|---|---|---|
| `st_regression` | `0.0` | `39.99999999988` | `40` (`leq`) | satisfied, margin 40, zero row | **active — exactly on the bound** |
| `spherical_tokamak_eval` | `0.0` | `40.2816` | `40` (`leq`) | satisfied, margin 40, zero row | **violated by +0.7%** |

`st_regression` is the one that matters: c56 is the single most binding constraint of
that problem and the port was solving a strictly relaxed version of it without one.
`spherical_tokamak_eval` runs in evaluation mode (`i_process_run_mode = -2`), so its
inequalities are reported and not driven — the freeze changed what the row *said*, not
where the answer went.

Registered as a seventh slot of `.tokamak.physics` (`psep_over_r_metric`), unswitched,
so an instance default — `Physics.run` computes it outside every `if`. Tier 1;
`test_exhaust.py::TestPsepOverRMetric` diffs the real staticmethod at *both* files'
converged answers, green plain and under `--fp-gradients` (82 passed in the file). No
cycle created and no SCC moved: `Blocking.scc` over both spherical tokamaks' graphs is
identical before and after (measured, not assumed — see `optimise_design.md` §29.3).
