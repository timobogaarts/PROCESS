# `process/core/init.py` — what it writes, and what the port must carry

**Audit only, 2026-08-31, closed.** No code was changed. Settles `next_steps_archive.md`
§22.4's estimate ("`init.py` is 1,302 lines and contains no physics") and scopes the
provider-side work of reproducing it. Measured by a stage-by-stage `DataStructure` diff
(`deepcopy` before/after each of `parse_input_file`, `set_active_constraints`,
`set_device_type`, `st_init`, `check_process`) across all seven `provider.CONFIGURATIONS`.

## Verdict

**"No physics" is right about computation, wrong about content — and the wrong half is
the half the port needs.** Not one of `init.py`'s 35 written fields has a physics formula
on its right-hand side (only literals, copies, `abs(x)>0`, `max(a,b)`, one `* 1.001`,
one arithmetic count) — so it is not a physics file. But four kinds of content in it
cannot be reproduced by a plain defaults table:

1. **A literature material-property table** switch-keyed on superconductor/insulation
   type (Nb3Sn 32 GPa, Bi-2212 80 GPa, NbTi 6.8 GPa, REBCO 145 GPa conductor axial
   Young's modulus; 20 GPa ITER / 2.5 GPa Kapton insulation; 0.13 ITER / 0.40 cryo-Al
   cryoplant efficiency) — the same shape as `dcond[]`, which has already produced one
   near-miss wrong number elsewhere in this port (`low_aspect_ratio_DEMO` escaped only
   because `dcond[4] == dcond[0]`).
2. **Three build-geometry identities** under the double-null branch
   (`dz_fw_plasma_gap = dz_xpoint_divertor`, `dz_shld_upper = dz_shld_lower`,
   `dz_vv_upper = dz_vv_lower`) — one of which overwrites a value the user set in the
   file. These belong to a build node, not a parse step.
3. **One optimiser bound moved from a physics comparison**: `boundl[3] =
   temp_plasma_pedestal_kev * 1.001` when the pedestal temperature is an iteration
   variable — changes the feasible region of the solve, not a default. Measured firing
   on `large_tokamak_nof`.
4. **Two physical consistency rules**: superconducting PF coil → `rho_pf_coil = 0`; no
   NBI → `f_nd_beam_electron = 0`.

`init.py` also holds 51 `raise ProcessValidationError` and 16 `logger.warning`/`error`
sites inside `check_process`, several encoding physics-validity ranges (e.g. water
undefined below 273.15 K, LTS conductor below 10 K). The port has no counterpart to any
of them — out of scope for this audit, not nothing.

One correction: an earlier record credited the stellarator's zeroed solenoid and
rewritten pulse times to `init.py`. They are `st_init`
(`process/models/stellarator/initialization.py`, called from `init_process`), a
different file with its own 18-field write set (§ below).

## Classification of the 35 fields

No genuine parse-time input is among them (those all arrive earlier, from
`parse_input_file`) — the interesting fact is the inverse: **three writes destroy a
genuine input** under the double-null branch. Of the 35: **8 resolve a sentinel default**
(a dataclass default that is not a value — `eff_tf_cryo`'s `-1.0`, `i_tf_wp_geom`'s
`UNSET`, etc. — a naive defaults table would read these as answers); **4 are presence
flags**, `True` only if the IN.DAT *named* a partner field, which no genuine PROCESS
input can ever set directly (this category holds the live defect below); **18 are
derivations** (a parse-time rule, e.g. double-null → `n_divertors=2`, or an alias copy
`f_nd_impurity_electron_array[i] = f_nd_impurity_electrons[i]` — the single largest
contributor to the `derived` boundary rows); **4 are physics-conditioned** zeroings.
**12 of 35 have a dataclass default that is not an answer.**

## Live defect found (still open, not fixed here)

`indat.py`'s factory reads `i_f_dr_tf_plasma_case`/`tfc_sidewall_is_fraction` as if they
were ordinary IN.DAT switches (`switches.get("tfc_sidewall_is_fraction", 0)`), but
**neither name is a declared PROCESS input** — `process/core/input.py`'s registry has no
such key, so the lookup can only ever return the fallback `0`. PROCESS itself sets both
from presence (`init.py:925-930`: `True` whenever the partner field is unset), which is
the case on **4 of the 7 reference configurations**
(`stellarator_helias`, `helias_5b`, `spherical_tokamak_eval`, `st_regression`) — so the
factory silently takes the wrong arm on those four. This is the cause of one of two
missing-producer defects recorded elsewhere (`st_regression` sets
`f_dr_tf_plasma_case` and comments out `dx_tf_side_case_min`; the port's
`dx_tf_side_case_min` slot resolves to `None`, no node, `boundary` reports it missing).
Fix: resolve both flags from the IN.DAT's *name set* (`provider.named_in` already
collects this) the way `init.py` itself does, not from a switch value that structurally
cannot exist.

## `st_init` — 18 fields, all forced whenever `istell != 0`

Distinct from `init.py`: build/PF (5, incl. `dr_cs → 0`, `iohcl → 0`), physics (4, incl.
`beta_norm_max → 0`, `q95 → 1.03`), current drive (1), times (7, incl.
`t_plant_pulse_burn → 3.15576e7`, one year), solver (1, `boundu[0] → 40.0`). Nine
overwrite a value the stellarator IN.DAT set explicitly. Measured: on tokamaks these same
boundary paths are answered correctly by `input`/`default`/`solver` *only because
`st_init` never runs there* — the identical provider code gets several of them wrong on a
stellarator, which is configuration-dependence at field level, not device-dependence in
the provider's own logic.

## Piecemeal absorption already in the port — and one of six is wrong

Grep found **six** `init.py`/`st_init` rules already independently re-implemented
somewhere in `functional_process/` (divertor count, TF shape default, TF winding-pack
geometry default, two SC-vs-resistive TF defaults, and the stellarator pedestal-zeroing
constant) — each with its own comment, none registered anywhere as a named `init.py`
boundary, no shared home. **One of the six is wrong**: the presence-flag defect above.
The pattern, not any one instance, is the finding: the same source file keeps getting
silently rediscovered.

## What this scopes (not done here — analysis only)

1. Fix the presence-flag defect above (§ Live defect) — closes one of two known
   missing-producer cases.
2. Give the provider an explicit `init_rule` layer for the bounded, enumerable set: the
   12 sentinel/presence defaults and 18 derivations. Acceptance test: the boundary's
   `off`-classified rows (13 across the seven pins, all traced to `init.py`/`st_init`)
   go to zero with no new disagreement.
3. The material-property table is not a provider concern — it's switch-keyed
   engineering data and belongs wherever `dcond[]` ends up living; the provider only
   needs to stop answering it from a stale dataclass default.
4. Treat `st_init` as a sixth arm of the stellarator variant dispatch, not a parse step —
   `ST_INIT_I_PLASMA_PEDESTAL` is the existing precedent for stating one of its 18 rules
   this way; seventeen more remain unstated.
5. `initialise_imprad` (`process/main.py:430`) is a **fifth** initialisation source (after
   parse, `init.py`, `st_init`, `check_process`), owning 4 `derived` rows on every pin by
   reading impurity-radiation data tables off disk — not a defaults problem, needs its
   own home.
6. The 51 raises / 16 warnings in `check_process` are out of scope but not nothing —
   roughly a dozen encode physics-validity ranges the port silently does not enforce.
