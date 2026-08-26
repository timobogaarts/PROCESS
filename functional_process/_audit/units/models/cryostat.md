---
kind: model-unit
status: draft
confidence: medium
---

**Ported (partial — minimal closure only).** `cryostat.py` / `test_cryostat.py`:
`calculate_r_cryostat_inboard`, tier-1, the sole occupant of `.tokamak.cryostat`.
Produces `.fwbs.r_cryostat_inboard`, the one variable `tokamak_boundary.md` lists this
slot as needing (read by `.buildings.sizing`). The other five fields
`Cryostat.external_cryo_geometry` computes are UNPORTED — see below.

## source

`process/models/cryostat.py`, 133 lines, 2 entered functions (`run`, 16-23;
`external_cryo_geometry`, 25-85) plus `output` (87-133, pure reporting, out of scope).
**Not** the stellarator's cryostat, which is a different model in a different file
(`process/models/stellarator/stellarator.py:1282-1330`, unit #1 chunk S5, already
ported, already a slot of `.stellarator.fwbs`) — flagging the distinction explicitly per
the wave-1 brief.

Only `external_cryo_geometry`'s first two lines (39-41) are in scope — see
§ scope discipline below.

## data footprint

| VarPath | read/write | classification | note |
|---|---|---|---|
| `.pf_coil.r_pf_coil_outer` | read | explicit-arg | `cryostat.py:40` — fixed-size array, `pfcoil_variables.py:366` (`np.zeros(NGC2)`); unported upstream (`.tokamak.pf_coil` is empty), so a boundary array input here |
| `.fwbs.dr_pf_cryostat` | read | explicit-arg | `:40` |
| `.fwbs.r_cryostat_inboard` | write | explicit-arg | `:39` |

Not ported (computed by the same method, immediately downstream of
`r_cryostat_inboard`, but not on this pass's boundary and not needed to produce it):

| VarPath | read/write | note |
|---|---|---|
| `.blanket.dz_pf_cryostat` | write | `:45-49`, reads `.build.f_z_cryostat` and the just-written `r_cryostat_inboard` |
| `.fwbs.z_cryostat_half_inside` | write | `:53-55`, reads `.pf_coil.z_pf_coil_upper` and `dz_pf_cryostat` |
| `.buildings.dz_tf_cryostat` | write | `:58-60`, reads `.build.z_tf_inside_half`, `.build.dr_tf_inboard` |
| `.fwbs.vol_cryostat_internal` | write | `:63-68` |
| `.fwbs.vol_cryostat` | write | `:71-80`, reads `.build.dr_cryostat` |
| `.fwbs.dewmkg` | write | `:83-85`, reads `.fwbs.vol_vv`, `.fwbs.den_steel` — **this is the same field `structure.md` records as an external read of `.tokamak.structure`'s `Structure` occupant** (`dewmass` there); since it is UNPORTED here, it stays a genuine boundary input for that unit until/unless a future pass extends this file's closure |

## scope discipline

`tokamak_boundary.md`'s `.tokamak.cryostat` row lists exactly one read:
`.fwbs.r_cryostat_inboard`. `external_cryo_geometry` computes it as a straight-line
sequence of six field writes, each depending on the previous; `r_cryostat_inboard` is
the *first* and needs nothing downstream of it. Per the wave-1 brief's scope discipline
("port the minimal closure of functions that produces your slot's listed output
variables"), only the two-line formula producing it is ported. The other five fields —
including `dewmkg`, which `structure.md` shows is read by a different unit already
in this pass — are UNPORTED, one-line reason: not on `.tokamak.cryostat`'s declared
boundary and not needed to compute what is.

## proposed signature(s)

```python
def calculate_r_cryostat_inboard(r_pf_coil_outer, dr_pf_cryostat) -> float:
```

## cottax node

`Cryostat(ExplicitFunction)`, in `functional_process/models/cryostat.py`. Owns
`.fwbs.r_cryostat_inboard`; reads `.pf_coil.r_pf_coil_outer`, `.fwbs.dr_pf_cryostat`.

## tier signal

**Tier 1.** No iteration, no calls into another model, no CoolProp, no switches.

**Sample provenance.** `tests/unit/models/test_cryostat.py::test_external_cryo_geometry`
provides one legacy point generated from `large_tokamak_eval.IN.DAT` itself, including a
22-element `r_pf_coil_outer` array and `dr_pf_cryostat = 0.5`, with
`expected_r_cryostat_inboard = 17.805470903073743` — used verbatim as this unit's legacy
sample (`max(r_pf_coil_outer) = 17.305470903073743`, `+ 0.5 = 17.805470903073743`,
confirms the formula independently of the reference call).

## switches touched

None.

## calls into other models

None.

## JAX-difficulty flags

None. `jnp.max` over a fixed-size array traces and differentiates cleanly (one-hot
gradient at the arg-max element, standard `jnp.max` behaviour, not a `safe_*` case).

## open questions

None.
