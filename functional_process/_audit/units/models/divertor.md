---
kind: model-unit
status: draft
confidence: medium
---

**Ported (partial by switch).** `divertor.py` / `test_divertor.py`: two occupants,
`calculate_divertor_heat_flux_split` (unconditional) and
`calculate_divertor_heat_load_wade` (`i_div_heat_load == WADE`, `n_divertors == 1`),
both tier-1. Together they produce both variables `tokamak_boundary.md` lists for
`.tokamak.divertor`: `.divertor.pflux_div_heat_load_mw` and
`.fwbs.p_div_nuclear_heat_total_mw`.

**Not** `process/models/stellarator/divertor.py` — that is a different model of a
different device's divertor, already ported (registry unit #4) and half of the
stellarator graph's one non-`problem` cycle. Recording the distinction per the wave-1
brief.

## source

`process/models/divertor.py`, 495 lines. In scope: `Divertor.run()` (29-104),
`Divertor.single_divertor_angle` (106-113), `Divertor.divwade` (272-406). Out of scope:
`Divertor.divtart` (115-270, `PENG_CHAMBER` arm, UNPORTED — different device class, tight
aspect ratio ST), `Divertor.incident_radiation_power`/`output()` (pure reporting plus a
sibling of `incident_neutron_power` whose output is not on this pass's boundary — see
§ scope discipline), `LowerDivertor`/`UpperDivertor` (457-495, double-null upper/lower
split subclasses — not reached; `n_divertors == 1` on the reference run, see §
switches touched).

## data footprint

`Divertor.run()`'s unconditional preamble (`:41-50`):

| VarPath | read/write | classification | note |
|---|---|---|---|
| `.blanket.deg_blkt_inboard_poloidal_plasma` | read | explicit-arg | `:113`, via the `single_divertor_angle` property |
| `.divertor.deg_div_poloidal_plasma` | write | explicit-arg | `:41` |
| `.fwbs.f_ster_div_single` | write | explicit-arg | `:42-44` — also read by `.tokamak.first_wall`'s `apply_first_wall_coverage_factors` occupant (this pass's other unit); a genuine intra-pass shared edge |
| `.physics.p_plasma_neutron_mw` | read | explicit-arg | `:47` |
| `.divertor.n_divertors` | read | explicit-arg (arithmetic) | `:49` — a plain multiplier here (`incident_neutron_power`'s third argument), not a branch; contrast its role in `divwade` below |
| `.fwbs.p_div_nuclear_heat_total_mw` | write | explicit-arg | `:46-50` |

Not ported (same preamble, sibling call, not on `.tokamak.divertor`'s boundary):

| VarPath | read/write | note |
|---|---|---|
| `.physics.p_plasma_rad_mw` | read | `:53` |
| `.fwbs.p_div_rad_total_mw` | write | `:52-56`, via `incident_radiation_power` (already a pure `@staticmethod`, trivially portable if ever needed — dropped only because nothing on this pass's boundary reads it) |

`Divertor.divwade` (`:272-406`), single-null (`n_divertors == 1`) arm only:

| VarPath | read/write | classification | note |
|---|---|---|---|
| `.physics.rmajor` | read | explicit-arg | `:274` |
| `.physics.rminor` | read | explicit-arg | `:275` |
| `.physics.aspect` | read | explicit-arg | `:276` |
| `.physics.b_plasma_toroidal_on_axis` | read | explicit-arg | `:277` |
| `.physics.b_plasma_poloidal_average` | read | explicit-arg | `:278` (source parameter name `b_plasma_poloidal_average`, matching the `DataStructure` field) |
| `.physics.p_plasma_separatrix_mw` | read | explicit-arg | `:279` |
| `.divertor.f_div_flux_expansion` | read | explicit-arg | `:280` |
| `.physics.nd_plasma_separatrix_electron` | read | explicit-arg | `:281` |
| `.divertor.deg_div_field_plate` | read | explicit-arg | `:282` |
| `.physics.rad_fraction_sol` | read | explicit-arg | `:283` |
| `.divertor.n_divertors` | read | switch (branch) | `:377` — see § switches touched; **this occupant does not take it as a parameter at all**, the single-null formula is baked in |
| `.physics.f_p_div_lower` | **not read** | — | only reached inside the `n_divertors == 2` branch, UNPORTED here — see § switches touched |
| `.divertor.pflux_div_heat_load_mw` | write | explicit-arg | `:380-382` |

## scope discipline

`.fwbs.p_div_rad_total_mw` is dropped even though `incident_radiation_power` (its
producer) is already a trivial, already-pure `@staticmethod` sitting right next to
`incident_neutron_power` — the one this record does port. Not ported because
`tokamak_boundary.md`'s `.tokamak.divertor` row lists only two reads and this is not one
of them; recorded explicitly so a future pass doesn't have to re-derive that it was a
choice, not an oversight.

## proposed signature(s)

```python
def calculate_divertor_heat_flux_split(
    deg_blkt_inboard_poloidal_plasma, p_plasma_neutron_mw, n_divertors,
) -> tuple[float, float, float]:  # deg_div_poloidal_plasma, f_ster_div_single, p_div_nuclear_heat_total_mw

def calculate_divertor_heat_load_wade(
    rmajor, rminor, aspect, b_plasma_toroidal_on_axis, b_plasma_poloidal_average,
    p_plasma_separatrix_mw, f_div_flux_expansion, nd_plasma_separatrix_electron,
    deg_div_field_plate, rad_fraction_sol,
) -> float:  # pflux_div_heat_load_mw, n_divertors == 1 baked in
```

## cottax node

`DivertorHeatFluxSplit(ExplicitFunction)` and `DivertorHeatLoadWade(ExplicitFunction)`,
in `functional_process/models/divertor.py`. Both proposed as sub-nodes of
`.tokamak.divertor` (registration is the consolidation pass's job — see report).

## tier signal

**Tier 1**, both occupants. No iteration, no calls into another model's method, no
CoolProp.

**Sample provenance.** `tests/unit/models/test_divertor.py::TestDivertor.test_divwade`
provides one legacy point for `divwade` with `f_p_div_lower = 1.0` — at that value the
`n_divertors == 2` formula (`max(f_p_div_lower * base, (1-f_p_div_lower) * base) =
max(base, 0) = base`) collapses exactly to the single-null formula this occupant ports,
so the legacy expected value (`0.58898578`) is reused verbatim as a legacy sample for
`calculate_divertor_heat_load_wade` even though the original test never set
`n_divertors` at all (it ran at the `DataStructure` default of `2`, coincidentally equal
here — confirmed algebraically, not just asserted).
`test_set_incident_neutron_power`'s parametrised cases are folded into
`calculate_divertor_heat_flux_split`'s legacy samples (`incident_neutron_power` is the
tail of that function; `deg_blkt_inboard_poloidal_plasma` is chosen so
`f_ster_div_single` matches each case's value directly).

## switches touched

| switch | reachable values | live on `large_tokamak_eval` | decision | evidence |
|---|---|---|---|---|
| `.divertor.i_div_heat_load` | `0` (`USER_INPUT`), `1` (`PENG_CHAMBER`), `2` (`WADE`) | `2` (`large_tokamak_eval.IN.DAT:139`) | **split** | Reads-sets are disjoint: `USER_INPUT` reads nothing (prints the existing value); `PENG_CHAMBER` (`divtart`) reads `triang`, `dz_xpoint_divertor`, `dr_fw_plasma_gap_inboard`, `i_single_null`, `dz_divertor`, `.tfcoil.drtop` — none of which `divwade` reads; `WADE` reads the ten fields in the data-footprint table above. `USER_INPUT`/`PENG_CHAMBER` UNPORTED |
| `.divertor.n_divertors` | `1` (single null), `2` (double null) | `1` — derived by `process/core/init.py:606-616` from `.physics.i_single_null = 1` (`large_tokamak_eval.IN.DAT:307`), **not** the `DataStructure` field's own default of `2` (`divertor_variables.py:94`, which only applies before `init.py` runs) | **split** (baked, no parameter) | `divwade`'s `:377-382`: `n_divertors == 2` reads `.physics.f_p_div_lower` and takes a `max` of two sub-terms; `n_divertors == 1` (else) reads nothing extra and returns `hldiv_base` directly. Genuinely different reads-set, so per the wave-1 policy this is an occupant-selecting switch, not a parameter — `calculate_divertor_heat_load_wade` has no `n_divertors` argument at all, it *is* the `n_divertors == 1` occupant. The `n_divertors == 2` arm is UNPORTED |

`i_div_heat_load == PENG_CHAMBER`'s `divtart` also reads `.physics.i_single_null` (as a
plain `DivertorNumberModels` selector for `areadv`'s single/double sum, its own internal
branch) — irrelevant here since `PENG_CHAMBER` is UNPORTED.

## calls into other models

None.

## JAX-difficulty flags

- `math.atan`/`math.degrees`/`math.radians`/`math.asin`/`math.cos`/`math.sin` (source)
  -> `jnp.arctan`/`jnp.degrees`/`jnp.radians`/`jnp.arcsin`/`jnp.sin` (port). Plain
  transcendental swaps, no domain restriction beyond `arcsin`'s `[-1, 1]` argument range
  — not flagged `safe_*` since the argument
  `(1 + 1/alpha_div**2) * sin(radians(deg_div_field_plate))` exceeds `1` whenever
  `alpha_div` is small (this is `divertor.md`'s own D1-equivalent, see § open questions)
  and PROCESS itself has no guard against it either (`math.asin` raises `ValueError`
  there; the port returns `nan`, matching `Tier1Contract`'s "domain error -> non-finite"
  convention with `reference_domain_errors = (ValueError,)`).
- **`safe_pow` applied to every fractional power in `calculate_divertor_heat_load_wade`**
  (`p_plasma_separatrix_mw**-0.02`, `b_plasma_poloidal_average**-0.92`, `aspect**0.42`,
  `(nd_plasma_separatrix_electron/1e19)**-0.02`, `p_plasma_separatrix_mw**-0.21`,
  `b_plasma_poloidal_average**-0.82`) -- found by `--fp-gradients`'s
  `test_gradient_finite_at_zero`, which failed at `aspect == 0`,
  `p_plasma_separatrix_mw == 0` and `nd_plasma_separatrix_electron == 0` before this fix
  (value finite, gradient `inf`/`nan` -- the exact `_audit/next_steps.md` §9 trap).
  `rmajor**0.04`/`rmajor**0.71` left as plain `**` since `rmajor` is never fuzzed to `0`
  (a major radius of zero is not a point any sample construction should reach) and the
  zero-boundary check does not probe it for that reason (it is not among this contract's
  `fuzz_bounds`-drawn arguments that can legally reach `0`... — actually `rmajor` *is* a
  `fuzz_bounds` argument here, but its lower bound (`2.0`) never draws `0`, and the
  boundary check only probes arguments actually in the sample, at exactly `0.0`; `rmajor`
  was not flagged by the run, confirming the point rather than assuming it).
- **`b_plasma_toroidal_on_axis == 0` is a genuine, unfixable-by-`safe_pow` division
  singularity**, found by the same check: `bp_omp/bt_omp` inside the `atan(...)` for
  `alpha_mid` divides by `bt_omp = -b_plasma_toroidal_on_axis * rmajor / r_omp`, which
  is exactly `0` there. `atan` saturates the resulting `+-inf` to a finite `+-pi/2`, so
  the *value* stays finite while the tangent through the division does not -- the same
  "unguarded division" class `_harness/boundary.py`'s existing register already carries
  several instances of (its own docstring: "a per-site modelling question, not a
  mechanical one"). Registered there
  (`("TestCalculateDivertorHeatLoadWade", "b_plasma_toroidal_on_axis")`) rather than
  worked around, per that file's own convention -- not fixed, not silently left failing.
- `divtart`'s `ProcessValueError` on `dz_xpoint_divertor <= 0.0` (`divertor.py:200-203`)
  is inside the UNPORTED `PENG_CHAMBER` arm; not relevant to this record's two occupants.

## open questions

- **`arcsin` domain.** `theta_div`'s argument can exceed `1` in magnitude for small
  `alpha_div` (large `f_div_flux_expansion` or small `alpha_mid`), which PROCESS reaches
  by raising `ValueError` from the C-level `math.asin` and the port reaches by returning
  `nan`. Not fixed, not worked around — flagged per `naming_convention.md`'s "domain
  guards are a declared expectation, not an escape." `large_tokamak_eval`'s own operating
  point is comfortably inside the valid domain (checked by the legacy sample passing).
- **Registration shape.** Whether `DivertorHeatFluxSplit`/`DivertorHeatLoadWade` should
  be two sibling sub-nodes of one `.tokamak.divertor` namespace, or two independently
  addressable slots, is left to the consolidation pass — see final report.
