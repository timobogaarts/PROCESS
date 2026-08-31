---
kind: model-unit
status: draft
confidence: high
---

**Not a numbered-node unit — it registers nothing.** One leaf pure function, called from
inside another unit's body, owning no `VarPath`. Same shape as `ivc_functions.md` and
`pumping.md`, and it has a registry row for the same reason: every record needs one.

Ported 2026-08-31, in the same pass as `pumping.md` — the two are one chain.

## source

`process/models/engineering/materials.py`, partial: `eurofer97_thermal_conductivity`
(lines 14-49) only.

**The other two functions in that file were already ported, elsewhere.**
`calculate_tresca_stress` (`materials.py:52-82`) and `calculate_von_mises_stress`
(`:85-129`) live in the port at `functional_process/models/pfcoil/stresses.py:313` and
`:334`, with `models/tfcoil/stress.py:862`'s `tresca_stress` a second transcription of
the Tresca formula. They landed in the stress packages because their callers are stress
models and the shared-helper convention had not reached this file. **They were not
re-ported here**: one formula with two homes is worse than the asymmetry. Whoever
consolidates the stress packages should lift them into
`functional_process/models/engineering/materials.py` then; the port module's docstring
carries the note asking for it. Note also that PROCESS decorates both with
`@numba.njit`, which the port necessarily drops — a JAX trace is the point.

Also in that file and not ported: `poisson_steel`, a module-level float constant.

## reachability

One caller in all of `process/`: `models/fw.py:487`, inside `FirstWall.fw_temp` —
reached only via `.fwbs.i_p_coolant_pumping == 2` (`MECHANICAL`), which no tracked
regression input selects (`large_tokamak_eval.IN.DAT:172` sets `3`). **This unblocks no
configuration today**; it is arbitrary-`IN.DAT` support. See `pumping.md` § reachability
for why porting it does not lift `indat.py`'s refusal of that arm — CoolProp does, and
this function is not why.

Unlike `pumping.md`'s three, this one is *not* here by the two-caller rule. It is here
because it mirrors PROCESS's filing and belongs to the same dormant chain.

## write trace — every write against its next use

Run before porting; the full trace is in `pumping.md` § write trace, which covers the
whole `fw_temp` / `coolant_friction_pressure_drop` chain in one place. The part specific
to this function:

**`tkfw` has two consumers and one of them is dead.** `fw.py:487` binds
`tkfw = eurofer97_thermal_conductivity(...)`. It is read at `:534`
(`deltat_solid_1D`) and `:556` (`deltat_solid`). `deltat_solid_1D` is read nowhere but
`:668`, inside `if output:` — it is the "Model B" the source explicitly says is *"given
for comparison"* while *"Model C is used"*. So one live path, one report line. The
function itself stays live through `deltat_solid` → `tpeakfw`.

**Its `temp` argument closes a cycle.** `temp_k = (temp_fw_coolant_out +
temp_fw_peak) / 2` at `fw.py:476`, and `.fwbs.temp_fw_peak` is written from `fw_temp`'s
own return at `blanket_library.py:2333` — last pass's estimate, per the source comment,
resolved only by `Caller.call_models`'s Gauss-Seidel. **This function sits inside an
SCC**, which is the thing to know before budgeting a port of `fw_temp`.

## data footprint

Already pure in `process/` — a module-level `def` over two floats, no `self.data`.

| VarPath | read/write | classification | note |
|---|---|---|---|
| *(none)* | — | — | pure polynomial |

Worth noting what the *caller* supplies: `fw_th_conductivity` is `.fwbs.fw_th_conductivity`
and `temp` is derived from `.fwbs.temp_fw_coolant_out` and `.fwbs.temp_fw_peak`. Those
become ports of whatever node eventually owns `fw_temp`, not of this function.

## proposed signature

```python
def eurofer97_thermal_conductivity(*, temp, fw_th_conductivity)
```

Unchanged from source except keyword-only, per package convention.

## cottax node

**None.** A leaf called from inside an unported body; owns no `VarPath`.

## tier signal

**Tier 1.** A cubic polynomial: no iteration, no branch, no division by a computed
quantity, no fractional power. The port needs no `jnp` import at all — the expression
traces on whatever it is handed.

**Sample provenance.** `tests/unit/models/engineering/test_materials.py::
test_eurofer97_thermal_conductivity` supplies the one legacy point, verbatim, including
its `temp = 1900.0` — well past the ~800 K the fit is documented for. Kept rather than
tidied: PROCESS extrapolates without complaint and a port that agreed only inside the
documented range would not be the same function.

## switches touched

None.

## calls into other models

None.

## JAX-difficulty flags

None. No `safe_math` site exists in this function — recorded explicitly so a later
reader does not have to re-derive the absence.

## open questions

- **When the stress packages are consolidated, `calculate_tresca_stress` and
  `calculate_von_mises_stress` should move into this module** and this record should
  grow the sections for them. Until then, this record's `## source` is honestly partial
  in a way `ivc_functions.md`'s was not: the missing pieces are ported, just filed
  somewhere else.
