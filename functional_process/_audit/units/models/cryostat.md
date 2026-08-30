---
kind: model-unit
status: draft
confidence: medium
---

**Ported (partial — the first four of six fields).** `cryostat.py` / `test_cryostat.py`:
`calculate_r_cryostat_inboard` and `calculate_cryostat_vertical_clearances`, both
tier-1, together the sole occupant of `.tokamak.cryostat`. They produce
`.fwbs.r_cryostat_inboard` — the one variable `tokamak_boundary.md` lists this slot as
needing — plus `.blanket.dz_pf_cryostat`, `.fwbs.z_cryostat_half_inside` and
`.buildings.dz_tf_cryostat`. The last three landed 2026-08-30; the remaining two fields
(`.fwbs.vol_cryostat_internal`, `.fwbs.vol_cryostat`) and `.fwbs.dewmkg` are UNPORTED —
see below.

## source

`process/models/cryostat.py`, 133 lines, 2 entered functions (`run`, 16-23;
`external_cryo_geometry`, 25-85) plus `output` (87-133, pure reporting, out of scope).
**Not** the stellarator's cryostat, which is a different model in a different file
(`process/models/stellarator/stellarator.py:1282-1330`, unit #1 chunk S5, already
ported, already a slot of `.stellarator.fwbs`) — flagging the distinction explicitly per
the wave-1 brief.

Only `external_cryo_geometry`'s lines 39-60 are in scope — see § scope discipline below.

## data footprint

| VarPath | read/write | classification | note |
|---|---|---|---|
| `.pf_coil.r_pf_coil_outer` | read | explicit-arg | `cryostat.py:40` — fixed-size array, `pfcoil_variables.py:366` (`np.zeros(NGC2)`); unported upstream (`.tokamak.pf_coil` is empty), so a boundary array input here |
| `.fwbs.dr_pf_cryostat` | read | explicit-arg | `:40` |
| `.fwbs.r_cryostat_inboard` | write | explicit-arg | `:39` |

Ported 2026-08-30 (the vertical chain, `:43-60`):

| VarPath | read/write | classification | note |
|---|---|---|---|
| `.build.f_z_cryostat` | read | explicit-arg | `:46` — a genuine PROCESS input (`core/input.py:443`, default `4.268`), written nowhere in `process/models/`; a **new** boundary input of the tokamak machine, declared by this producer |
| `.fwbs.r_cryostat_inboard` | read | local-intermediate | `:47` — the value written two lines above, in the same straight-line method, with no branch between |
| `.blanket.dz_pf_cryostat` | write | explicit-arg | `:45-49` |
| `.pf_coil.z_pf_coil_upper` | read | explicit-arg | `:54` — fixed-size array, same provenance as `r_pf_coil_outer` |
| `.blanket.dz_pf_cryostat` | read | local-intermediate | `:54` |
| `.fwbs.z_cryostat_half_inside` | write | explicit-arg | `:53-55` |
| `.build.z_tf_inside_half` | read | explicit-arg | `:59` |
| `.build.dr_tf_inboard` | read | explicit-arg | `:59` |
| `.buildings.dz_tf_cryostat` | write | explicit-arg | `:58-60` — **an `InputVariable` this method overwrites**, see § scope discipline |

Not ported (computed by the same method, downstream of `dz_tf_cryostat`):

| VarPath | read/write | note |
|---|---|---|
| `.fwbs.vol_cryostat_internal` | write | `:63-68` |
| `.fwbs.vol_cryostat` | write | `:71-80`, reads `.build.dr_cryostat` |
| `.fwbs.dewmkg` | write | `:83-85`, reads `.fwbs.vol_vv`, `.fwbs.den_steel` — **this is the same field `structure.md` records as an external read of `.tokamak.structure`'s `Structure` occupant** (`dewmass` there), and it is on `missing_producers_tokamak.txt` in its own right. It needs `vol_cryostat`, which needs `.build.dr_cryostat`: its own closure, left for the pass that takes it rather than folded in unaudited |

## scope discipline

**Wave 1's rule, and why it did not hold.** `tokamak_boundary.md`'s `.tokamak.cryostat`
row lists exactly one read: `.fwbs.r_cryostat_inboard`. `external_cryo_geometry` computes
it as a straight-line sequence of six field writes, each depending on the previous;
`r_cryostat_inboard` is the *first* and needs nothing downstream of it. Per the wave-1
brief's scope discipline ("port the minimal closure of functions that produces your
slot's listed output variables"), only the two-line formula producing it was ported.

That rule is per-slot and the defect it missed is per-machine:
`.buildings.dz_tf_cryostat` is read by `.buildings.sizing`
(`process/models/buildings.py:245`) and was owned by nobody, so the graph served it from
the `DataStructure` — and, unlike the other nineteen holes of the same wave, **not as a
`0.0`**. It is a real PROCESS `InputVariable` (`core/input.py:377`, default `2.5`,
`buildings_variables.py:56`), so the frozen value was `2.5` against PROCESS's
`5.5730055` on `large_tokamak_nof`: a plausible number, not a suspicious one.

**Establishing that it is a producer and not an input** is the part worth recording,
because "PROCESS writes it" is not by itself enough (`tokamak_boundary.md` §"Five with
no producer anywhere on the traced surface" makes the opposite call for other fields).
Three measurements, all of which had to hold:

1. The write is **unconditional** — `external_cryo_geometry:58-60` has no branch above
   it, and `caller.py:351` calls `Cryostat.run()` on every tokamak pass.
2. The write happens **before** every live read — `caller.py:351` (cryostat) precedes
   `caller.py:370` (buildings), the only live reader.
3. Every *other* read of the field in `process/` is inside an `if output:` reporting
   block (`models/build.py:191-219`, `:456-512`, `:769-776`). Nothing computes with the
   input value.

`large_tokamak_nof.IN.DAT` does not set the field at all, so on that run the seed is the
dataclass default and is dead twice over. The three lines producing it are therefore
ported, and the two intermediates on the way (`dz_pf_cryostat`,
`z_cryostat_half_inside`) with them — PROCESS writes those too, so
`boundary.computed_by_process` counts them, and `costs_2015.py` already reads
`z_cryostat_half_inside` on a cost model this machine does not select.

## proposed signature(s)

```python
def calculate_r_cryostat_inboard(r_pf_coil_outer, dr_pf_cryostat) -> float:

def calculate_cryostat_vertical_clearances(
    f_z_cryostat, r_cryostat_inboard, z_pf_coil_upper, z_tf_inside_half, dr_tf_inboard,
) -> tuple[float, float, float]:  # dz_pf_cryostat, z_cryostat_half_inside, dz_tf_cryostat
```

Two functions rather than four, because the last three PROCESS statements are
consecutive, unconditional and each reads the one before — there is nothing between them
for a caller to choose. Same idiom as `calculate_divertor_heat_flux_split`.

## cottax node

`Cryostat(ExplicitFunction)`, in `functional_process/models/cryostat.py`. Owns
`.fwbs.r_cryostat_inboard`, `.blanket.dz_pf_cryostat`, `.fwbs.z_cryostat_half_inside`
and `.buildings.dz_tf_cryostat`; reads `.pf_coil.r_pf_coil_outer`,
`.fwbs.dr_pf_cryostat`, `.build.f_z_cryostat`, `.pf_coil.z_pf_coil_upper`,
`.build.z_tf_inside_half`, `.build.dr_tf_inboard`.

## tier signal

**Tier 1**, both functions. No iteration, no calls into another model, no CoolProp, no
switches.

**Sample provenance.** `tests/unit/models/test_cryostat.py::test_external_cryo_geometry`
provides one legacy point generated from `large_tokamak_eval.IN.DAT` itself, including a
22-element `r_pf_coil_outer` array and `dr_pf_cryostat = 0.5`, with
`expected_r_cryostat_inboard = 17.805470903073743` — used verbatim as this unit's legacy
sample (`max(r_pf_coil_outer) = 17.305470903073743`, `+ 0.5 = 17.805470903073743`,
confirms the formula independently of the reference call).

That same generated point carries `f_z_cryostat`, `z_pf_coil_upper`,
`z_tf_inside_half`, `dr_tf_inboard` and all four `expected_*` values, so the vertical
chain's contract reuses it and nothing about this unit's samples is hand-built. Its
`r_cryostat_inboard` argument takes the point's `expected_r_cryostat_inboard`, because
the stored `r_cryostat_inboard = 0` is the pre-call value the method overwrites.

**The vertical chain's reference has one adapter and it is exact.** The port's chain
starts from `r_cryostat_inboard`, which `external_cryo_geometry` computes rather than
accepts, so the reference drives PROCESS to that value with one PF coil at
`r_cryostat_inboard` and `dr_pf_cryostat = 0.0`. `np.max` of a one-element array is that
element and `x + 0.0 == x` for every finite double, so no rounding is introduced between
the port's argument and PROCESS's own first line.

## switches touched

None.

## calls into other models

None.

## JAX-difficulty flags

None. `jnp.max` over a fixed-size array traces and differentiates cleanly (one-hot
gradient at the arg-max element, standard `jnp.max` behaviour, not a `safe_*` case).

## open questions

None.
