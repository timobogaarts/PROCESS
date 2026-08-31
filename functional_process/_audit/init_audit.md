# `process/core/init.py` — what it writes, and what the port must carry

**Status:** audit only, 2026-08-31. No code was changed. This record settles
`next_steps.md` §22.4's flagged estimate ("`init.py` is 1,302 lines and contains no
physics") and scopes the work of cutting the PROCESS seed on the input side.

Every claim below is marked **[measured]** or **[read]**. *Measured* means a stage-by-stage
diff of the `DataStructure` over all seven `provider.CONFIGURATIONS`: `parse_input_file`,
`set_active_constraints`, `set_device_type`, `st_init` and `check_process` were each
wrapped with a `deepcopy` before and after, inside a real `SingleRun`, and the resulting
field diffs recorded. *Read* means read off the source of `init.py` and not exercised by
any of the seven files.

---

## 1. The verdict on §22.4

**"No physics" is right about computation and wrong about content, and the wrong half is
the half the port needs.**

**[measured]** Not one write in `init.py` has a physics formula on its right-hand side.
The complete set of right-hand-side shapes across all 35 fields it writes is: a literal
constant; a copy of another field; `abs(x) > 0`; `a + b + 1` on two integers; `max(a, b)`;
`teped * 1.001`; and `num_constraints - n_*`. There is no model, no closure, no scaling.
To that extent §22.4 stands and `init.py` is not a physics file.

**But "no physics" was being used to license "so it is just defaults", and that does not
follow.** Four kinds of content in `init.py` cannot be carried by a defaults table:

1. **A literature material-property table** (`:961-1034`, `:933-940`). Insulation Young's
   modulus (20 GPa ITER design / 2.5 GPa Kapton), conductor axial Young's modulus keyed on
   the superconductor (Nb3Sn 32 GPa, Bi-2212 80 GPa, NbTi 6.8 GPa, REBCO 145 GPa, each with
   its DOI in a comment), and cryoplant efficiency (0.13 ITER / 0.40 Strawbridge
   extrapolation for cryo-Al). This is engineering data selected by a switch. It is exactly
   the shape of `dcond[]`, which has already produced one near-miss wrong number in this
   port (`next_steps.md` §14.5's `CoilsMass`: `low_aspect_ratio_DEMO` assembled the
   `dcond[0]` occupant and escaped only because `dcond[4] == dcond[0]`). Three of these
   fields are `off` rows on every tokamak pin today.
2. **Three build-geometry identities** under the double-null branch (`:610-612`):
   `dz_fw_plasma_gap = dz_xpoint_divertor`, `dz_shld_upper = dz_shld_lower`,
   `dz_vv_upper = dz_vv_lower`. These belong to a build node, not to a parse step, and one
   of them overwrites a value the user set in the file.
3. **One optimiser bound moved from a physics comparison** (`:456-458`):
   `boundl[3] = temp_plasma_pedestal_kev * 1.001` when `temp_plasma_electron_vol_avg_kev`
   is an iteration variable. **[measured]** this fires on `large_tokamak_nof`. It changes
   the feasible region of the solve, not a default.
4. **Two physical consistency rules** (`:1140`, `:1145-1147`): a superconducting PF coil has
   `rho_pf_coil = 0`, and a machine with no NBI has `f_nd_beam_electron = 0`. Both are `off`
   rows on every tokamak pin.

**[measured]** `init.py` additionally holds **51 `raise ProcessValidationError` sites and 16
`logger.warning`/`error` sites**, all inside `check_process`, and several encode
physics-validity ranges (water cannot flow below 273.15 K; the cryo-Al resistivity fit is
undefined above 40–50 K; LTS conductor below 10 K; the NSTX confinement scaling is for
A < 1.7). The port has no counterpart to any of them. They are the "fails loudly" half and
are out of scope here, but they are not nothing.

**Restated claim, safe to build on:** *`init.py` computes no physics, but it resolves
sentinels, carries a switch-keyed material-property table, derives build geometry, and
moves a solver bound. A defaults table reproduces none of those four.*

**One correction of attribution.** §22.6 credited the stellarator's zeroed solenoid and
rewritten pulse times to `init.py`. **[measured]** they are `st_init`, in
`process/models/stellarator/initialization.py`, which `init_process` *calls* — 18 fields,
listed in §4. The distinction matters for the port: those writes live under `models/`, and
`indat.ST_INIT_I_PLASMA_PEDESTAL` already names that file as their source.

---

## 2. Classification of every write

**[measured]** `init.py` writes **35 distinct `DataStructure` fields**. None of them is a
genuine parse-time input — genuine inputs arrive from `parse_input_file`, which runs before
any of this. Category 3 of the brief is therefore **empty**, and the interesting fact is the
inverse: **three writes destroy a genuine input** (`dz_shld_upper`, `dz_vv_upper`,
`dz_fw_plasma_gap`, all under the double-null branch).

### 2a. Sentinel resolution — 8 fields

A dataclass default that is *not a value*. This is the category where a naive defaults table
supplies a confident wrong number.

| field | default | resolves to | line | fired on |
|---|---|---|---|---|
| `.tfcoil.eff_tf_cryo` | `-1.0` | `0.13` SC / `0.40` cryo-Al | `:933-940` | **7/7** [measured] |
| `.tfcoil.i_tf_bucking` | `-1` | `0` Cu / `1` otherwise | `:891-895` | 5/7 [measured] |
| `.tfcoil.i_tf_wp_geom` | `-1` (`UNSET`) | `0`/`1` from `i_tf_turns_integer` | `:977-989` | 5/7 [measured] |
| `.tfcoil.i_tf_shape` | `0` (`DEFAULT`) | picture-frame ST / D-shape | `:728,:775` | **7/7** [measured] |
| `.tfcoil.i_cp_joints` | `-1` | `0` SC / `1` resistive | `:752-756` | 0/7 [read] |
| `.tfcoil.eyoung_ins` | `1e8` | `20e9` / `2.5e9` | `:961-975` | **7/7** [measured] |
| `.tfcoil.eyoung_cond_axial` | `6.6e8` | `0` / `32e9` / `80e9` / `6.8e9` / `145e9` | `:992-1027` | **7/7** [measured] |
| `.numerics.n_equality_constraints` | `-1` | `count - n_inequality` | `:1285-1294` | **7/7** [measured] |

`eyoung_ins` and `eyoung_cond_axial` are sentinels of a subtler kind: the default is a
plausible-looking number (`1e8` Pa, `6.6e8` Pa) that `init.py` treats as "unset" and
replaces by two orders of magnitude. A defaults table reads them as answers.

### 2b. Presence flags — 4 fields

**A boolean recording whether the IN.DAT *named* a field.** Not one of the four is a
declared PROCESS input, so no input file can ever set them and the dataclass `False` is
never a user's choice.

| field | test | line | `True` on |
|---|---|---|---|
| `.tfcoil.tfc_sidewall_is_fraction` | `dx_tf_side_case_min < 1e-11` | `:925-926` | 4/7 [measured] |
| `.tfcoil.i_f_dr_tf_plasma_case` | `dr_tf_plasma_case < 1e-11` | `:929-930` | 4/7 [measured] |
| `.tfcoil.i_dx_tf_turn_general_input` | `abs(dx_tf_turn_general) > 0` | `:1074` | 2/7 [measured] |
| `.tfcoil.i_dx_tf_turn_cable_space_general_input` | `abs(dx_..._cable_space_general) > 0` | `:1104` | 0/7 [measured] |

This category is structurally distinct from all the others: the answer is not a function of
any *value*, it is a function of the **set of names the file contains**. `provider.named_in`
already collects exactly that set, which is the one piece of machinery this needs.

**This category contains a live defect — see §3.**

### 2c. Derivation — a node or a parse-time rule — 18 fields

| field | rule | line | measured |
|---|---|---|---|
| `.divertor.n_divertors` | `2` if double-null else `1` | `:609,:617` | 5/7 |
| `.build.dz_fw_plasma_gap` | `= dz_xpoint_divertor` (DN) | `:610` | 2/7 |
| `.build.dz_shld_upper` | `= dz_shld_lower` (DN) | `:611` | 2/7 — **overrides an input** |
| `.build.dz_vv_upper` | `= dz_vv_lower` (DN) | `:612` | 0/7 [read] |
| `.buildings.esbldgm3` | `0.0` if not pulsed | `:827` | 3/7 |
| `.build.dr_blkt_inboard` | `0.0` if no inboard blanket | `:1152` | 0/7 [read] |
| `.tfcoil.n_tf_stress_layers` | `i_tf_bucking + n_tf_graded_layers + 1` | `:920` | **7/7** |
| `.pf_coil.i_pf_location[0:3]` | forced for `itart=1, itartpf=0` | `:641-643` | 0/7 [read] |
| `.tfcoil.temp_cp_average` | `= temp_cp_coolant_inlet` (cryo-Al) | `:711` | 0/7 [read] |
| `.impurity_radiation.f_nd_impurity_electron_array[i]` | `= f_nd_impurity_electrons[i]` | `:381-384` | **7/7** — a pure alias |
| `.buildings.triv`, `.heat_transport.p_tritium_plant_electric_mw` | `0.0` if tritium fraction negligible | `:370-371` | 0/7 [read] |
| `.tfcoil.temp_tf_superconductor_margin_min`, `.temp_cs_...` | `= tmargmin` (deprecated alias) | `:1189-1190` | 4/7, 2/7 |
| `.tfcoil.eyoung_cond_trans` | `0` or `= eyoung_cond_axial` | `:996,:1031-1034` | **7/7** |
| `.globals.icase` | a label string | `:621,:825,:1300-1302` | **7/7** — output only |
| `.numerics.active_constraints`, `.n_inequality_constraints` | from `icc` | `:1279-1294` | **7/7** — problem statement |

`.impurity_radiation.f_nd_impurity_electron_array` deserves its own note: the declared input
is `f_nd_impurity_electrons` (`input.py:198`), a *different field*. `init.py` copies it into
a second array which nothing declares. That single four-line loop is the largest single
contributor to the pins' `derived` rows (§5).

### 2d. Physics-conditioned — 4 fields

| field | rule | line | measured |
|---|---|---|---|
| `.pf_coil.rho_pf_coil` | `0.0` if the PF conductor is superconducting | `:1140` | **7/7** |
| `.physics.f_nd_beam_electron` | `0.0` if there is no NBI | `:1145-1147` | **7/7** |
| `.physics.temp_plasma_electron_vol_avg_kev` | `= temp_plasma_pedestal_kev * 1.001` | `:440` | 0/7 [read] |
| `.numerics.boundl[3]` / `.boundu[3]` | raised to `teped * 1.001` | `:456-458` | 1/7 (`large_tokamak_nof`) |

### 2e. Count

| category | fields | of which sentinel-defaulted |
|---|---|---|
| sentinel resolution | 8 | 8 |
| presence flags | 4 | 4 (a `False` no input can set) |
| derivation | 18 | 1 (`n_tf_stress_layers`, default `0`) |
| genuine parse-time input | **0** | — |
| physics | 0 formulas; 4 fields written under a physics-shaped condition | 0 |

**12 of 35 fields have a dataclass default that is not an answer.**

---

## 3. A live defect the audit found

**`.tfcoil.tfc_sidewall_is_fraction` and `.tfcoil.i_f_dr_tf_plasma_case` are read by the
port's factory from the IN.DAT's switch scan, and neither is a declared PROCESS input.**

`indat.py:4420-4428`:

```python
    dr_tf_plasma_case=_slot_occupant(
        "i_f_dr_tf_plasma_case",
        bool(switches.get("i_f_dr_tf_plasma_case", 0)),  # `tfcoil_variables.py:83`
        DR_TF_PLASMA_CASE,
    ),
    dx_tf_side_case_min=_slot_occupant(
        "tfc_sidewall_is_fraction",
        bool(switches.get("tfc_sidewall_is_fraction", 0)),  # `:95`
        DX_TF_SIDE_CASE_MIN,
        build=lambda cls: None if cls is None else cls(),
    ),
```

**[measured]** Neither name appears in `process/core/input.py`'s registry, so
`switches.get(...)` can only ever return the fallback `0`. `init.py:925-930` sets both to
`True` whenever the partner field is unset — which is **4 of the 7 configurations**
(`stellarator_helias`, `helias_5b`, `spherical_tokamak_eval`, `st_regression`). The factory
takes the `False` arm where PROCESS takes the `True` arm.

**[measured]** This is the cause of one of §22.6's two new missing producers.
`st_regression.IN.DAT:1000` has `dx_tf_side_case_min` commented out and sets
`f_dr_tf_plasma_case` instead; PROCESS therefore computes `dx_tf_side_case_min` from the
fraction, while the port's `dx_tf_side_case_min` slot resolves to `None` — no node, nothing
produces it, and `boundary` reports a missing producer. `large_tokamak_eval.IN.DAT:369-370`
sets both fields explicitly, which is why the tokamaks never saw it.
`models/tfcoil/base.py:41-46`'s "`tfc_sidewall_is_fraction == False` gets no node at all …
`large_tokamak_eval.IN.DAT` does not set `tfc_sidewall_is_fraction`" is correct about
`large_tokamak_eval` and wrong as a general statement: no IN.DAT can set it.

`.build.r_cp_top`, §22.6's other missing producer, is **not** an `init.py` field —
`init.py:758-764` only validates that `i_r_cp_top == 1` when it is set. Its cause is
elsewhere.

---

## 4. `st_init` — 18 fields, all on `istell != 0`

**[measured]**, identical on both stellarators:

- **Build/PF (5):** `.build.dr_cs → 0`, `.build.dr_cs_tf_gap → 0`, `.build.iohcl → 0`,
  `.build.f_dr_tf_outboard_inboard → 1.0`, `.pf_coil.f_z_cs_tf_internal → 0`.
- **Physics (4):** `.physics.beta_norm_max → 0`, `.kappa95 → 1.0`, `.triang → 0`,
  `.q95 → 1.03`. (`.physics.i_plasma_pedestal → 0` is written too but both files already
  hold `0`, so no diff — `indat.ST_INIT_I_PLASMA_PEDESTAL` already pins this one.)
- **Current drive (1):** `.current_drive.i_hcd_calculations → 0`.
- **Times (7):** three ramp times → `0`, `.t_plant_pulse_burn → 3.15576e7` (one year), and
  the three sums `t_plant_pulse_plasma_present` / `_no_burn` / `_total` recomputed from them.
- **Solver (1):** `.numerics.boundu[0] → 40.0` (aspect ratio upper bound).

Nine of these overwrite a value the stellarator IN.DAT set explicitly (e.g.
`t_plant_pulse_burn` `1000 → 3.15576e7`, `dr_cs` `0.811 → 0`). **[measured]** 5–6 of them
are boundary paths on every configuration's pin, and on the tokamaks they are answered
`input`/`default`/`solver` and are right *only because `st_init` does not run there* — the
same provider code gets four of them wrong on a stellarator. That is the configuration
dependence §22.2 asserted, seen at field level.

---

## 5. Accounting against the pins

### 5a. Every `derived` row is explained, and `init.py` owns 11–13 of each

**[measured]** by matching each pin's `derived` rows against the measured stage write sets.

| configuration | `derived` | `init.py` | `initialise_imprad` | unexplained |
|---|---|---|---|---|
| `stellarator_helias` | 17 | 13 | 4 | **0** |
| `helias_5b` | 17 | 13 | 4 | **0** |
| `large_tokamak_nof` | 16 | 12 | 4 | **0** |
| `large_tokamak_eval` | 17 | 13 | 4 | **0** |
| `low_aspect_ratio_DEMO` | 16 | 12 | 4 | **0** |
| `spherical_tokamak_eval` | 16 | 12 | 4 | **0** |
| `st_regression` | 15 | 11 | 4 | **0** |

`init.py`'s share is `f_nd_impurity_electron_array[i]` (11 or 12 element paths, all from the
four-line alias loop at `:381-384`) plus `.divertor.n_divertors` where the machine is
single-null. The remaining 4 on every configuration are the impurity data tables
(`impurity_arr_zav`, `m_impurity_amu_array`, `pden_impurity_lz_nd_temp_array`,
`temp_impurity_keV_array`), written by `initialise_imprad` from `process/main.py:430` — a
**fifth** initialisation source, not `init.py` and not `st_init`, and one nothing in §22 has
named. It reads data files from disk.

### 5b. All 13 `off` rows are `init.py` or `st_init`, in a 7/6 split

**[measured]**, one behaviour per row, with no row left over:

| `off` path | source | rule |
|---|---|---|
| `.tfcoil.eff_tf_cryo` | `init.py:933-940` | sentinel `-1.0` → `0.13` |
| `.tfcoil.eyoung_ins` | `init.py:961-975` | sentinel `1e8` → `2e10` |
| `.tfcoil.eyoung_cond_axial` | `init.py:992-1034` | material table / `0` |
| `.pf_coil.rho_pf_coil` | `init.py:1140` | SC → `0` |
| `.physics.f_nd_beam_electron` | `init.py:1145-1147` | no NBI → `0` |
| `.buildings.esbldgm3` | `init.py:827` | not pulsed → `0` |
| `.build.dz_shld_upper` | `init.py:611` | double-null → `= dz_shld_lower`, over an input |
| `.build.dr_cs` | `st_init:23` | stellarator → `0` |
| `.build.dr_cs_tf_gap` | `st_init:26` | stellarator → `0` |
| `.times.t_plant_pulse_burn` | `st_init:45` | → `3.15576e7` |
| `.times.t_plant_pulse_coil_precharge` | `st_init:43` | → `0` |
| `.times.t_plant_pulse_plasma_current_ramp_up` | `st_init:44` | → `0` |
| `.times.t_plant_pulse_plasma_current_ramp_down` | `st_init:46` | → `0` |

**7 `init.py`, 6 `st_init`.** §22.6's "13 distinct boundary paths … end up at a value neither
the input file nor the dataclass default supplies" is confirmed and now fully attributed.

### 5c. The latent count is larger than the `off` count

**[measured]** Counting boundary paths whose base field `init.py` *can* write:

| configuration | in boundary | `off` today | `derived` | `solver` | **latent** |
|---|---|---|---|---|---|
| `stellarator_helias` / `helias_5b` | 9 | 1 | 2 | 1 | **5** |
| `large_tokamak_nof` / `_eval` / `low_aspect_ratio_DEMO` | 16 | 5 | 2 | 1 | **8** |
| `spherical_tokamak_eval` / `st_regression` | 15 | 5 | 1 | 1 | **8** |

*Latent* = a path the provider answers independently, whose value `init.py` would move on a
different switch setting, and which agrees today only because the branch did not fire —
`.buildings.triv`, `.heat_transport.p_tritium_plant_electric_mw`, `.build.dr_blkt_inboard`,
`.build.dz_vv_upper`, `.build.dz_fw_plasma_gap`, `.build.dz_shld_upper` (on the machines
where it is single-null), `.tfcoil.eyoung_cond_trans`, `.buildings.esbldgm3` (where pulsed).
Add `st_init`'s 5–6 per pin, which are latent on every tokamak by the same argument.

So the pins' 13 `off` rows are a *lower bound* on `init.py`'s reach, as §22.6 said — and the
measured upper bound on the seven configurations is roughly **twice** that.

---

## 6. What the port has already absorbed piecemeal

**[measured]** by grep over `functional_process/`. Six `init.py`/`st_init` rules are
re-implemented in the port, each independently, each with its own comment saying so, and
none of them registered anywhere as an `init.py` boundary:

| port site | rule reproduced | source |
|---|---|---|
| `indat.py:2400` `_n_divertors` | `i_single_null → n_divertors` | `init.py:606-617` |
| `indat.py:2491` `_tf_shape` | `i_tf_shape` `DEFAULT` → picture-frame / D-shape | `init.py:728-729, 775-776` |
| `indat.py:2517` `_tf_wp_geom` | `i_tf_wp_geom` `UNSET` → from `i_tf_turns_integer` | `init.py:977-989` |
| `indat.py:3170` `_tf_field_and_force_arm` | `i_cp_joints == -1 → 0` for SC | `init.py:752-756` |
| `indat.py:3197` `_tf_stress_arm` | `i_tf_bucking == -1 → 1` for SC | `init.py:891-895` |
| `indat.py:2005` `ST_INIT_I_PLASMA_PEDESTAL` | `i_plasma_pedestal → 0` on a stellarator | `st_init:31` |

**And one rule the port absorbed *incorrectly*** — §3 above: `indat.py:4420-4428` reads two
`init.py`-derived presence flags as if they were IN.DAT switches, and is wrong on 4 of 7
configurations.

A further set of sites *document* the dependence without re-implementing it, i.e. they
consume the seed's already-resolved value and say so:
`models/divertor.py:17-19`, `models/shield.py:13,97`, `models/fw.py:30`,
`models/build.py:1654,1724`, `models/namespace.py:212`, `indat.py:1307` (the `i_pf_location`
overwrite, in a refusal reason), `indat.py:1354` and `indat.py:671`.

**The pattern is the finding.** Six separate absorptions, six separate rediscoveries of the
same file, no shared home, no registry row, and one of them wrong. The provider is the
natural place to put them: it already asks "who supplied this value, and why", and
`init.py` is the answer for 11–13 `derived` rows and 7 `off` rows per configuration.

---

## 7. What this scopes

Ordered by what the evidence says costs least and buys most.

1. **Fix `indat.py:4420-4428`** — resolve the two presence flags from the IN.DAT's *name
   set* the way `init.py:925-930` does, rather than from a switch value that cannot exist.
   Expected: one of the two ST missing producers closes. (Not done here — analysis only.)
2. **Give the provider an `init_rule` layer.** Twelve sentinel-or-presence defaults
   (§2a, §2b) and eighteen derivations (§2c) are a bounded, enumerable table, not a rewrite.
   The `off` rows are the acceptance test: all 13 must go to zero without a new
   disagreement appearing.
3. **The material table (§1.1) is not a provider concern** — `eff_tf_cryo`, `eyoung_ins`,
   `eyoung_cond_axial/trans` are switch-keyed engineering data and belong wherever
   `dcond[]` ends up living. Deciding that is `§14.5`'s question, not this one; the provider
   only needs to stop answering them from a stale dataclass default.
4. **`st_init` is a sixth arm of the stellarator variant dispatch, not a parse step.** Its
   18 writes are `istell != 0` forcings, five to six of which are boundary paths on every
   pin. `ST_INIT_I_PLASMA_PEDESTAL` is the precedent for how to state one; there are
   seventeen more.
5. **`initialise_imprad` (`main.py:430`) is a fifth initialisation source** and owns 4
   `derived` rows on every pin. It reads data files from disk, so it is not a defaults
   problem at all; it is the impurity radiation tables, and it will need a home.
6. **The 51 raises and 16 warnings are out of scope but are not nothing.** They are the
   "parser fails loudly" property §22.4 credited to `input.py`, and roughly a dozen of them
   encode physics-validity ranges the port silently does not enforce.
