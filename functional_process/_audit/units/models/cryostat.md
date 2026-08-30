---
kind: model-unit
status: draft
confidence: medium
---

**Fully ported as of 2026-08-30.** `cryostat.py` / `test_cryostat.py`:
`calculate_r_cryostat_inboard` and `calculate_external_cryo_geometry`, tier-1, one node
at `.tokamak.cryostat` owning all **seven** fields `Cryostat.external_cryo_geometry`
writes. (This record said "six" throughout, counting `dz_tf_cryostat` and
`vol_cryostat_internal` as one stage of the dependency chain -- they are written from
the same predecessor. Six stages, seven fields, and a node's outputs are counted in
fields.)

Only the first of the seven was ported on 2026-08-26, and the § scope discipline section
below is that decision, left standing. § the other six, ported (2026-08-30) is what
overturned it and why — briefly: the rule that stopped at one field asks what *this
slot's* boundary lists, and a field whose only reader is in another subsystem is
invisible to that question.

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

The other six, ported 2026-08-30 (computed by the same method, immediately downstream
of `r_cryostat_inboard`; the "not ported" heading this table carried until then is the
decision § the other six, ported reverses):

| VarPath | read/write | note |
|---|---|---|
| `.blanket.dz_pf_cryostat` | write | `:45-49`, reads `.build.f_z_cryostat` and the just-written `r_cryostat_inboard` |
| `.fwbs.z_cryostat_half_inside` | write | `:53-55`, reads `.pf_coil.z_pf_coil_upper` and `dz_pf_cryostat` |
| `.buildings.dz_tf_cryostat` | write | `:58-60`, reads `.build.z_tf_inside_half`, `.build.dr_tf_inboard` |
| `.fwbs.vol_cryostat_internal` | write | `:63-68` |
| `.fwbs.vol_cryostat` | write | `:71-80`, reads `.build.dr_cryostat` |
| `.fwbs.dewmkg` | write | `:83-85`, reads `.fwbs.vol_vv`, `.fwbs.den_steel` — **this is the same field `structure.md` records as an external read of `.tokamak.structure`'s `Structure` occupant** (`dewmass` there). It is no longer a boundary input for that unit: this file's closure was extended on 2026-08-30, exactly as the sentence that stood here anticipated |

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

## the other six, ported (2026-08-30)

**The scope rule above was applied correctly and still gave the wrong answer**, and the
gap is worth naming because it is a property of the rule and not of this unit.
"Port the minimal closure that produces your slot's listed output variables" asks what
*this slot's* readers want, and `tokamak_boundary.md` derives that list from reads made
inside the graph. A field whose only reader lives in another subsystem is invisible to
that question -- and both of the fields that mattered here are exactly that shape:

* `.fwbs.dewmkg` is read by `models/structure.py::StructureMasses` for
  `.structure.coldmass`. PROCESS computes `1.44e7` kg for it on `large_tokamak_nof`; the
  port was reading `0.0`.
* `.buildings.dz_tf_cryostat` is read by the site-buildings sizing, same story.

`boundary.unproduced_but_computed` is what could see them, because it asks PROCESS's own
write set rather than the graph's read set, and both were on the eighteen-row missing
producer list it produced on 2026-08-30. The extension costs nothing structural: the
whole method is one straight-line sequence with no branch anywhere in it, so the seven
fields are one node exactly as they are one method, and every read it adds
(`.build.f_z_cryostat`, `.build.dr_cryostat`, plus `.pf_coil.z_pf_coil_upper`,
`.build.z_tf_inside_half`, `.fwbs.vol_vv`, `.fwbs.den_steel`) is either a genuine
`IN.DAT` input or already owned. The guess count on the tokamak boundary is unchanged,
so it closed no loop either.

## proposed signature(s)

```python
def calculate_r_cryostat_inboard(r_pf_coil_outer, dr_pf_cryostat) -> float:

def calculate_external_cryo_geometry(
    r_pf_coil_outer, dr_pf_cryostat, f_z_cryostat, z_pf_coil_upper,
    z_tf_inside_half, dr_tf_inboard, dr_cryostat, vol_vv, den_steel,
) -> tuple  # the seven fields, in the source's own order
```

`calculate_r_cryostat_inboard` is kept as a function of its own and called by the
second: it is separately Tier-1 tested, and PROCESS's first line is genuinely a
sub-expression of the rest rather than an artefact of the earlier scoping.

## cottax node

`Cryostat(ExplicitFunction)`, in `functional_process/models/cryostat.py`. Owns all six:
`.fwbs.r_cryostat_inboard`, `.blanket.dz_pf_cryostat`, `.fwbs.z_cryostat_half_inside`,
`.buildings.dz_tf_cryostat`, `.fwbs.vol_cryostat_internal`, `.fwbs.vol_cryostat`,
`.fwbs.dewmkg`. The three areas are PROCESS's own scattering from one method, not a
choice made here.

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

None. The one this record used to imply -- whether the other six fields should be
ported -- was answered on 2026-08-30 by the measure that could see the question, and is
recorded above rather than left open.
