---
kind: model-unit
status: draft
confidence: high
---

## source
`process/models/stellarator/stellarator.py`, lines 2457-2600: `st_phys_output()`. Chunk
1G of unit #1 (see `../../_audit/unit_registry.md`). This is the last method in the file
(2600 lines total) — the read range runs to EOF, nothing truncated.

## data footprint

All 20 parameters (`q_PROCESS`, `total_q_neo_e`, `dmdt_neo_fuel_from_e`, `q_PROCESS_r1`,
`chi_PROCESS_e`, `chi_neo_e`, `q_neo_e`, `g_neo_e`, `dndt_neo_e`, `rho_ne_max`,
`rho_te_max`, `gradient_length_ne`, `gradient_length_te`, `rho_star`, `nu_star_e`,
`nu_star_D`, `nu_star_T`, `nu_star_He`, `nd_plasma_electron_line`,
`nd_plasma_electrons_max`) are **plain explicit function arguments already** — no
`self.data.*` access anywhere in the method body. Only other attribute used is
`self.outfile` (an output-file handle, `constants.NOUT` — not a `DataStructure` field, no
`VarPath`).

| VarPath | read/write | classification | note |
|---|---|---|---|
| *(none)* | — | — | zero `self.data.*` reads or writes in this method. All inputs arrive as parameters; nothing is written back to any field. |

No implicit-io, no implicit-io-via-callee, no redundant-duplicate-write instances — the
four-way classification simply doesn't apply here, there's nothing to classify.

## proposed signature(s)

**None proposed — out of scope.** This is a pure reporting shell (`po.oheadr`/`po.ovarre`
calls writing to `self.outfile`), not a computation. Per the project's general policy that
`Model.output()`-style methods carry no computational role, this method is not a
candidate for porting to the functional codebase at all, unlike `density_limits.py`'s
`output()` (see below) which turned out to have a real computation hiding inside it.

## tier signal

**N/A — not computational.** Confirmed by full read, not assumed (see directive). Three
inline ratios are computed for display only (`q_PROCESS / q_PROCESS_r1`,
`gradient_length_te / gradient_length_ne`, `nd_plasma_electron_line /
nd_plasma_electrons_max`) — trivial arithmetic on already-supplied parameters, not stored
anywhere, not fed back into any state, and not needed by anything downstream. This is a
materially different situation from `density_limits.py`'s `output()`
(`functional_process/_audit/units/models/stellarator/density_limits.md`), which called
`power_at_ignition_point` — a multi-step computation (including the deep-copy + hardcoded
double-solve) — to produce values that exist *only* for that report. Here, nothing is
computed that the caller doesn't already hand in.

**Conclusion for the directive's question: this chunk is purely reporting.** No
computation-invoking pattern found, unlike the precedent in `density_limits.py`.

## switches touched
None.

## calls into other models
None — only `process.core.process_output` helpers (`po.oheadr`, `po.ovarre`), which are
output-formatting utilities, not models.

## JAX-difficulty flags
None applicable — out of pure-functional-port scope entirely (see "proposed signatures").

## open questions
None.
