---
kind: model-unit
status: draft
confidence: medium
---

**Partially ported (2026-08-26), see "## ported" below.** The subset live on
`tests/regression/input_files/large_tokamak_eval.IN.DAT` -- `calculate_minor_radius`,
`calculate_shape_ipdg89_x_point` (`i_plasma_geometry == 0` only), `calculate_geometry_
double_arc`, plus functions 3-9 verbatim (including `sauter_geometry`, ported but not
yet wired to an occupant class). No `unit_registry.md` row, no `next_steps.md` edit;
registration is the consolidation pass's job (`next_steps.md` §4b) -- see "## ported"'s
registration instructions.

**Record path note, for the consolidation pass to settle.** `schema.md` says a record
lives "at the path mirroring its source file within `functional_process/`". The source is
`process/models/physics/plasma_geometry.py`, so the mirroring path is
`_audit/units/models/physics/plasma_geometry.md` — alongside `confinement_time.md`,
`composition.md` and the rest of the `models/physics/` records. This file was written to
`_audit/units/models/plasma_geometry.md` because that is the path the task named. What
binds a record to its unit is its `unit_registry.md` row, not adjacency, so either path
works mechanically; flagging so the divergence is a decision rather than an accident.

## source

`process/models/physics/plasma_geometry.py` (1229 lines, full file in scope). This is the
tokamak plasma-geometry model and the sole site of `i_plasma_geometry`, one of the 17 new
topology decisions in `_audit/tokamak_scope.md`. It is the **first** model
`Caller._call_models_once` runs on the tokamak path (`process/core/caller.py:284`), ahead
of `build` and `physics`.

**23 `def`s.** Ten are structure, not computation: nine `IntEnum` plumbing methods
(`__new__` ×3, `full_name`, `description`, `kappa_model`, `triang_model`,
`kappa95_model`, `triang95_model`) plus `PlasmaGeom.__init__` (sets `self.outfile` only).
**Thirteen are in audit scope:**

| # | function | lines | shape |
|---|---|---|---|
| 1 | `PlasmaGeom.run` | 202–511 | the stateful shell; 13-way `i_plasma_geometry` dispatch + 2 more switches |
| 2 | `PlasmaGeom.output` | 513–709 | reporting shell — but writes one `data` field and calls real computation-free enum lookups that raise |
| 3 | `PlasmaGeom.plasma_angles_arcs` | 711–759 | `@staticmethod`, pure |
| 4 | `PlasmaGeom.plasma_poloidal_perimeter` | 761–783 | `@staticmethod`, pure |
| 5 | `PlasmaGeom.plasma_surface_area` | 785–830 | `@staticmethod`, pure |
| 6 | `PlasmaGeom.plasma_volume` | 832–896 | `@staticmethod`, pure |
| 7 | `PlasmaGeom.plasma_cross_section` | 898–931 | `@staticmethod`, pure |
| 8 | `PlasmaGeom.sauter_geometry` | 933–1001 | `@staticmethod`, pure |
| 9 | `PlasmaGeom.calculate_iter_physics_basis_elongation` | 1003–1032 | `@staticmethod`, pure — **not called by `run()`**; sole caller is `models/physics/confinement_time.py:203` |
| 10 | `surfa` | 1040–1085 | module-level, pure, **dead in `process/`** |
| 11 | `perim` | 1088–1126 | module-level, pure, **dead in `process/`** |
| 12 | `fvol` | 1129–1182 | module-level, pure, **dead in `process/`** |
| 13 | `xsect0` | 1185–1229 | module-level, pure, **dead in `process/`** |

"Dead in `process/`" is measured: `grep -rn "surfa(\|perim(\|fvol(\|xsect0(" process/`
returns nothing outside this file; the only importers are
`tests/unit/models/physics/test_plasma_geom.py`. See defect **D11** — two of the four are
algebraically identical to the live pair.

## the extraction seam

**Unusually clean, and the cleanest of any tokamak model looked at so far.** Functions
3–9 are already `@staticmethod`s taking plain floats and returning plain floats/tuples —
they are `CallableNode.fn` already, needing only `np.` → `jnp.` and the `safe_pow`
treatment (§Derivative-safe power laws below). **Zero `self.data` access inside any of
them.**

The seam is therefore exactly at the `run()` boundary, and the work is one-sided: every
`data` read and write in the file is in `run()` (`202–511`) and `output()`, and every
piece of arithmetic that is not the `i_plasma_geometry` dispatch is already outside them.

The one thing `run()` does that is *not* a read/write shell is the 13-branch
`i_plasma_geometry` dispatch itself (lines 231–433, ~200 lines) — that arithmetic is
inline, has no `calculate_*` extraction, and is the actual porting work in this file.
Each branch is 2–6 lines of straight-line algebra; there is no shared body between
branches at all (see §switches touched, which matters for the split-vs-static policy
question `traceability_policy.md` leaves open).

## data footprint

Reference run: `tests/regression/input_files/large_tokamak_eval.IN.DAT` —
`i_plasma_geometry = 0`, `i_plasma_current = 4`, `i_plasma_shape` unset (default `0`),
`i_plasma_wall_gap` unset (default `1`). Rows marked *(live)* are on that path.

### `run()` — unconditional preamble (lines 216–227)

| VarPath | read/write | classification | note |
|---|---|---|---|
| `.physics.rmajor` | read | explicit-arg | *(live)* never written by any model in tokamak mode (grep: only `evaluators.py` logging) — a pure boundary input / iteration variable (ixc 3) |
| `.physics.aspect` | read | explicit-arg | *(live)* likewise a boundary input (ixc 1); only `stellarator.py:220` writes it, and that path returns before `plasma_geom` runs |
| `.physics.rminor` | **write** | explicit-arg | *(live)* `rmajor / aspect`. This file is the **sole tokamak producer** of `rminor`; read back at lines 439/440/445/453/477 within the same call |
| `.physics.eps` | **write** | explicit-arg | *(live)* `1 / aspect`. Sole tokamak producer. Read back at lines 248/252/255 (branch 1 only) |

`xsi = xso = thetai = thetao = xi = xo = 0.0` (lines 216–221) are dead initialisers —
every one is unconditionally overwritten at 445/453 before use. Cosmetic (**D9**).

### `run()` — the `i_plasma_geometry` dispatch (lines 231–433)

This is the load-bearing table. **Which of `kappa`/`triang`/`kappa95`/`triang95` is a
read and which is a write changes per branch** — this is
`conditional-ownership-by-run-config` in its strongest form yet recorded: not "owned
unless it is in `ixc`" (`build.md`'s `dr_blkt_*`, `st_new_config`'s `aspect`) but "owned
or read depending on a plain input switch".

| `i_plasma_geometry` | enum member | reads (beyond `aspect`) | writes |
|---|---|---|---|
| 0 *(live)* | `IPDG89_X_POINT` | `.physics.kappa`, `.physics.triang` | `.physics.kappa95`, `.physics.triang95` |
| 1 | `STAR_FIESTA` | `.physics.eps` | `.physics.q95_min`, `.kappa`, `.triang`, `.kappa95`, `.triang95` |
| 2 | `ZOHM_ITER_X_POINT` | `.physics.fkzohm`, `.aspect`, `.triang` | `.kappa`, `.kappa95`, `.triang95` |
| 3 | `ZOHM_ITER_95` | `.physics.fkzohm`, `.aspect`, `.triang95` | `.kappa`, `.triang`, `.kappa95` |
| 4 | `IPDG89_95` | `.physics.kappa95`, `.triang95` | `.kappa`, `.triang` |
| 5 | `MAST_DATA_95` | `.physics.kappa95`, `.triang95` | `.kappa`, `.triang` |
| 6 | `MAST_DATA_X_POINT` | `.physics.kappa`, `.triang` | `.kappa95`, `.triang95` |
| 7 | `FIESTA_RUNS_95` | `.physics.kappa95`, `.triang95` | `.kappa`, `.triang` |
| 8 | `FIESTA_RUNS_X_POINT` | `.physics.kappa`, `.triang` | `.kappa95`, `.triang95` |
| 9 | `INDUCTANCE_SCALING_X_POINT` | `.physics.ind_plasma_internal_norm`, `.aspect`, `.triang` | `.kappa`, `.kappa95`, `.triang95` |
| 10 | `CREATE_DATA_EU_DEMO_X_POINT` | `.physics.aspect`, `.m_s_limit`, `.triang` | `.kappa95`, `.kappa`, `.triang95` |
| 11 | `MENARD_2016_X_POINT` | `.physics.aspect`, `.triang` | `.kappa`, `.kappa95`, `.triang95` |
| 12 | `MENARD_1997_X_POINT` | `.physics.aspect`, `.triang` | `.kappa`, `.kappa95`, `.triang95` |

Classification for the intra-branch reads of a just-written field (branch 2 writes
`kappa` at 272 then reads it at 277; branch 3 the same; branches 9/11/12 likewise; branch
10 writes `kappa95` at 377 then reads it at 389/390/391/393/396) — **`local-intermediate`**
in every case: straight-line, unconditional-within-the-branch, no intervening call. They
become plain Python locals in a per-branch port. The one exception is branch 10's
`kappa95`, whose read-back is *inside* an `if` on its own value (line 389) — still
`local-intermediate` as to origin, but see **F3** in the JAX flags.

`.physics.kappa`/`.triang` read at line 445 (into `plasma_angles_arcs`) is
**`implicit-io`** at the `run()` level and cannot be classified without fixing the
switch: it is either the value the caller supplied (branches 0/6/8) or the value this
same call computed 200 lines earlier (branches 1–5, 7, 9–12). Splitting per branch makes
it an explicit graph edge in both cases; keeping one node makes it unrepresentable.

### `run()` — scrape-off layer (lines 438–440)

| VarPath | read/write | classification | note |
|---|---|---|---|
| `.physics.i_plasma_wall_gap` | read | switch | *(live, `=1`)* |
| `.physics.rminor` | read | local-intermediate | `i_plasma_wall_gap == 0` only |
| `.build.dr_fw_plasma_gap_inboard` | write (`==0`) / **not touched** (`==1`) | conditional-ownership-by-run-config | *(live: not touched, so a boundary input; the reference file supplies `0.25`)* |
| `.build.dr_fw_plasma_gap_outboard` | write (`==0`) / **not touched** (`==1`) | conditional-ownership-by-run-config | same |

Identical shape to `build.md`'s `dr_blkt_inboard`/`dr_blkt_outboard`, and the symmetric
case: when this node does not own them, they are read by `build` as external inputs, so
there *is* a clean "or it's an external input" story here (unlike `build.md`'s
`dz_shld_upper`, open question 2).

### `run()` — arcs and outboard area (lines 445–461, unconditional)

| VarPath | read/write | classification | note |
|---|---|---|---|
| `.physics.rminor` | read | local-intermediate | into `plasma_angles_arcs` and `plasma_surface_area` |
| `.physics.kappa` | read | implicit-io | see above — provenance is switch-dependent |
| `.physics.triang` | read | implicit-io | same |
| `.physics.rmajor` | read | explicit-arg | |
| `.physics.a_plasma_surface_outboard` | **write** | explicit-arg | `= xso`. Written **even on the Sauter path**, where it is the double-arc answer and `a_plasma_surface` is not — see **D10**. Consumed by `models/blankets/dcll.py:820,838` |

`xi`, `thetai`, `xo`, `thetao`, `xsi`, `xso` are Python locals, not `data` fields —
`local-intermediate` in spirit but with no `VarPath`, so not tabulated. `xsi` is
**computed and then discarded** on the Sauter path (**D10**).

### `run()` — the geometry-model arm (lines 467–509)

| VarPath | read/write | classification | note |
|---|---|---|---|
| `.physics.i_plasma_current` | read | switch | *(live, `=4`)* only the `== 8` test is used |
| `.physics.i_plasma_shape` | read | switch | *(live, `=0`)* |
| `.physics.plasma_square` | read | explicit-arg | **Sauter arm only** |
| `.physics.f_vol_plasma` | read | explicit-arg | **double-arc arm only** *(live)* — never assigned anywhere in `process/`, so a pure boundary input (default `1.0`, input range 0.001–10.0). See **D2** |
| `.physics.len_plasma_poloidal` | **write** | explicit-arg | *(live)* sole producer in `process/` for tokamak runs |
| `.physics.vol_plasma` | **write** | explicit-arg | *(live)* sole tokamak producer |
| `.physics.a_plasma_poloidal` | **write** | explicit-arg | *(live)* sole tokamak producer |
| `.physics.a_plasma_surface` | **write** | explicit-arg | *(live)* sole tokamak producer |

### `output()` (lines 513–709)

| VarPath | read/write | classification | note |
|---|---|---|---|
| `.stellarator.istell` | read | switch | gates almost the entire body |
| `.divertor.n_divertors` | read | switch | `0/1/2`, else `ProcessValueError` |
| `.physics.itart` | read | switch | `0`/`1`; no `else` — see **D8b** |
| `.physics.itart_r` | **write** | — (reporting artefact) | `= itart`, a float copy made solely so `po.ovarre` can print an integer. **A `data` write from a reporting function** — same class as `density_limits.md`'s "`output()` isn't purely reporting", though far more benign: nothing else in `process/` reads `itart_r` |
| `.physics.i_plasma_shape` | read | switch | `PlasmaShapeModelType(...)` raises `ValueError` off-range |
| `.physics.i_plasma_geometry` | read | switch | `PlasmaGeometryModelType(...)` raises `ValueError` off-range |
| `.physics.rmajor`, `.rminor`, `.aspect`, `.plasma_square`, `.kappa`, `.kappa95`, `.triang`, `.triang95`, `.fkzohm`, `.len_plasma_poloidal`, `.a_plasma_poloidal`, `.a_plasma_surface`, `.vol_plasma` | read | explicit-arg | pure reporting |
| `.physics.kappa_ipb` | read | **cross-model read** | produced by `models/physics/confinement_time.py:203` from `vol_plasma`/`rmajor`/`rminor` — i.e. from *this* node's own outputs, one model later. Reporting-only, so not an SCC edge in the compute graph, but it is why `calculate_iter_physics_basis_elongation` lives in this file at all |

## coupling / SCC finding

**`i_plasma_geometry == 9` closes a genuine two-node cycle**, confirmed by reading both
sides:

- `plasma_geometry.py:353` reads `.physics.ind_plasma_internal_norm` to compute `.physics.kappa`.
- `models/physics/physics.py:4743` **writes** `.physics.ind_plasma_internal_norm`
  (unless `i_ind_plasma_internal_norm == 0`, `USER_INPUT`), and `physics.py:4727`'s
  `calculate_internal_inductance_menard` **reads `.physics.kappa`** to do it.
- `physics.run()` is called *after* `plasma_geom.run()` (`caller.py:284` then `:290`).

So under `i_plasma_geometry == 9` **and** `i_ind_plasma_internal_norm != 0`, PlasmaGeom
consumes a value produced later in the same pipeline pass, and the producer consumes
PlasmaGeom's own output. Today this is closed only by `Caller.call_models`'s
"run everything up to 10× until idempotent" — nothing declares it. A second, weaker edge
runs through `.physics.vol_plasma` → `physics.py:4731`'s `ind_plasma_internal_norm_iter_3`
→ the same field, live whenever `i_ind_plasma_internal_norm` selects the ITER-3 model.

**Not live on `large_tokamak_eval`** (`i_plasma_geometry = 0`), and unconfirmed by
measurement — this is a read of the call graph, not an instrumented run. Recording it
because it is a candidate SCC that only exists for *one value of one switch*, which is
itself an argument for the split default: a union node would show this cycle
unconditionally, on every configuration, and 12 of 13 configurations do not have it.

## the enum is a machine-readable ownership table

`PlasmaGeometryModelType` (lines 63–193) carries, per value, four
`PlasmaGeometryModels` tags saying which published model each of `kappa`, `triang`,
`kappa95`, `triang95` came from. **Checked branch by branch against the code: all 13
rows are consistent**, and `USER_INPUT` in a slot corresponds exactly to that field
appearing in the *reads* column of the dispatch table above, never the writes. That makes
the enum a pre-existing, in-source statement of the per-value read/write split this audit
would otherwise have to derive — worth reusing directly when the graph is assembled, and
worth a regression check that it stays in step with the formulas.

One row is a labelling gap rather than an inconsistency: value 9's `kappa_model` is
`PlasmaGeometryModels.UNKNOWN` ("Unknown"), so `output()` prints
*"X-Point Elongation set from: Unknown"* for a branch that has a perfectly definite
formula (`kappa` from `ind_plasma_internal_norm` and `aspect`).

## proposed signature(s)

Functions 3–9 need **no signature change at all** — port them verbatim, `np.` → `jnp.`
plus `safe_pow`/`safe_sqrt`:

```python
def plasma_angles_arcs(a, kappa, triang) -> tuple[float, float, float, float]:  # xi, thetai, xo, thetao
def plasma_poloidal_perimeter(xi, thetai, xo, thetao) -> float
def plasma_surface_area(rmajor, rminor, xi, thetai, xo, thetao) -> tuple[float, float]   # xsi, xso
def plasma_volume(rmajor, rminor, xi, thetai, xo, thetao) -> float
def plasma_cross_section(xi, thetai, xo, thetao) -> float
def sauter_geometry(a, r0, kappa, triang, square) -> tuple[float, float, float, float]
def calculate_iter_physics_basis_elongation(vol_plasma, rmajor, rminor) -> float
```

New, extracted from `run()`'s preamble:

```python
def calculate_minor_radius(rmajor: float, aspect: float) -> tuple[float, float]:  # rminor, eps
```

The dispatch becomes **13 functions, one per `i_plasma_geometry` value**. The naming
convention's "derive the name from the `VarPath`s it owns" collides here with 13 branches
owning overlapping sets, so name them from the enum member instead — that name already
encodes both the model and which surface it is anchored on:

```python
def calculate_shape_ipdg89_x_point(kappa, triang) -> tuple[float, float]:            # kappa95, triang95
def calculate_shape_star_fiesta(eps) -> tuple[float, float, float, float, float]:   # q95_min, kappa, triang, kappa95, triang95
def calculate_shape_zohm_iter_x_point(fkzohm, aspect, triang) -> tuple[float, float, float]
def calculate_shape_zohm_iter_95(fkzohm, aspect, triang95) -> tuple[float, float, float]
def calculate_shape_ipdg89_95(kappa95, triang95) -> tuple[float, float]              # kappa, triang
def calculate_shape_mast_data_95(kappa95, triang95) -> tuple[float, float]
def calculate_shape_mast_data_x_point(kappa, triang) -> tuple[float, float]
def calculate_shape_fiesta_runs_95(kappa95, triang95) -> tuple[float, float]
def calculate_shape_fiesta_runs_x_point(kappa, triang) -> tuple[float, float]
def calculate_shape_inductance_scaling_x_point(ind_plasma_internal_norm, aspect, triang) -> tuple[float, float, float]
def calculate_shape_create_data_eu_demo_x_point(aspect, m_s_limit, triang) -> tuple[float, float, float]
def calculate_shape_menard_2016_x_point(aspect, triang) -> tuple[float, float, float]
def calculate_shape_menard_1997_x_point(aspect, triang) -> tuple[float, float, float]
```

The wall-gap and geometry-model arms, likewise split:

```python
def calculate_sol_gaps_from_rminor(rminor: float) -> tuple[float, float]   # i_plasma_wall_gap == 0 only
def calculate_geometry_double_arc(rmajor, rminor, kappa, triang, f_vol_plasma) -> tuple[float, float, float, float]
def calculate_geometry_sauter(rmajor, rminor, kappa, triang, plasma_square) -> tuple[float, float, float, float]
def calculate_a_plasma_surface_outboard(rmajor, rminor, kappa, triang) -> float   # unconditional, both arms
```

No `cottax node` section: per `schema.md`, that is skipped while open questions about the
signature are unresolved, and OQ1/OQ2/OQ4 below are exactly that.

## tier signal

**Tier 1 for all 13 in-scope functions.** No `scipy.optimize`, no `fsolve`, no ad hoc
fixed-iteration loop, no call into another `Model`'s method anywhere in the file, no
CoolProp, no external library beyond `numpy`. `run()` is a dispatch shell over pure
arithmetic; `output()` is a reporting shell (with the one `itart_r` write noted above).

This is the tier-1-est file audited so far: unlike `density_limits.md`, `output()` here
invokes no real computation, and unlike `build.md`, the pure cores already exist as
`@staticmethod`s rather than needing extraction from a straight-line `run()` body.

**Sample provenance is the weak point, not tier.** `tests/unit/models/physics/test_plasma_geom.py`
covers `plasma_angles_arcs`, `plasma_surface_area`, `plasma_volume`,
`plasma_cross_section`, `calculate_iter_physics_basis_elongation` and all four dead
legacy functions — but **not `sauter_geometry`, not `plasma_poloidal_perimeter`, not
`run`, not `output`**. And across every tracked regression input,
`i_plasma_geometry ∈ {0, 10}`, `i_plasma_current ∈ {4, 9}`, `i_plasma_shape` is never
set. **The Sauter arm has no regression oracle at all** and 11 of the 13 geometry models
have no converged operating point to sample from — fuzz within
`ITERATION_VARIABLES` bounds is the only option there, and see **D1** for why that is
dangerous in this particular file.

## switches touched

Seven, five of them in `run()`. Every one is **split**; the reads-set evidence is above.

| switch | reachable values | live on `large_tokamak_eval` | decision | evidence |
|---|---|---|---|---|
| `.physics.i_plasma_geometry` | 0–12 (input range `(0, 12)` inclusive; all 13 have enum members and code) | `0` | **split** — mandatory, and stronger than "reads differ" | reads differ across the 13 (table above), and so do *writes*: `kappa`/`triang`/`kappa95`/`triang95` swap between the read set and the write set. A union node would have to both own and read the same four `VarPath`s — not merely over-connected, **structurally inexpressible** in cottax |
| `.physics.i_plasma_wall_gap` | `0`, `1` (`choices=[0,1]`) | `1` | **split** | `0` reads `.physics.rminor` and owns `.build.dr_fw_plasma_gap_{inboard,outboard}`; `1` reads nothing and owns nothing. Conditional ownership, same shape as `build.md`'s `blktmodel` |
| `.physics.i_plasma_current` | 1–9; only `== 8` is tested here | `4` | **split** (compound, see below) | selects the Sauter arm |
| `.physics.i_plasma_shape` | `0`, `1` (`choices=[0,1]`) | `0` | **split** (compound, see below) | selects the Sauter arm |
| `.stellarator.istell` | `0` vs `!= 0` | `0` | **split** (reporting) | `output()` only; `!= 0` suppresses ~90% of the body. Same criterion as `stellarator_A_orchestration.md`'s `output` bool |
| `.divertor.n_divertors` | `0`, `1`, `2`, else raise | — | **identical reads** | `output()` only; selects a string literal, reads nothing else. A genuine static-kwarg case if reporting is ever ported |
| `.physics.itart` | `0`, `1` | `0` | **identical reads**, but writes | `output()` only; both arms write `.physics.itart_r` from the same source and differ only in the printed label. Static kwarg |

### the compound Sauter switch

Line 467: `if i_plasma_current == 8 or i_plasma_shape == PlasmaShapeModelType.SAUTER:`.
Two independent switches disjoined into one binary choice. Reads-sets:

- **Sauter arm**: `rmajor`, `rminor`, `kappa`, `triang`, `plasma_square`
- **double-arc arm**: `rmajor`, `rminor`, `kappa`, `triang` (via `xi/thetai/xo/thetao`), `f_vol_plasma`

Disjoint in `plasma_square` vs `f_vol_plasma` → **split**. The graph-assembly rule is
`i_plasma_current == 8 or i_plasma_shape == 1`, i.e. the two switches collapse to one
boolean *at build time* — which is the cleanest possible outcome and worth recording as
such: a compound switch does not have to become a compound node, it becomes one predicate
evaluated once by the assembler. It does, however, mean `i_plasma_current`'s own split
(driven by `plasma_current.py`) and this file's split must be resolved together.

### a measured contribution to the "79 invented edges" count

For the live configuration (`i_plasma_geometry = 0`) a single node declaring the union of
all 13 branches' reads would claim **9** incoming `VarPath` edges where the run uses
**2**: 7 invented edges from this one switch. Per branch the invented count runs 6–8. And
the four fields that swap sides (`kappa`, `triang`, `kappa95`, `triang95`) are not
countable as invented edges at all — a union node cannot express them, which is a
qualitatively worse failure than over-declaring.

### a concrete instance of `traceability_policy.md`'s open "shared remainder" clause

The policy notes six deliberate deviations from split-by-default and says the missing
clause is about *the size of the shared remainder*. This switch is a clean data point at
the opposite extreme from `coelc`'s `itart` (15 differing lines of 290): **the 13 branches
share literally no body.** Each is 2–6 lines, all of them differing, with zero common
tail. Splitting duplicates nothing.

More usefully, the branches fall into **reads-identical families**, and within a family
the policy's static-kwarg *exception* is satisfied on its own terms:

- **A = {0, 6, 8}** — read `kappa`, `triang`; write `kappa95`, `triang95`. Identical reads, three different fits (IPDG89 / MAST / FIESTA).
- **B = {4, 5, 7}** — read `kappa95`, `triang95`; write `kappa`, `triang`. Identical reads; the exact inverses of family A.
- **C = {11, 12}** — read `aspect`, `triang`; write `kappa`, `kappa95`, `triang95`. Identical reads (Menard 2016 / Menard 1997).
- Singletons: 1 (`eps`), 2 (`fkzohm`), 3 (`fkzohm`, `triang95`), 9 (`ind_plasma_internal_norm`), 10 (`m_s_limit`).

So the honest answer for this file is **five-plus-three: split into 8 nodes across
families and singletons, with a static kwarg selecting the fit inside families A, B and
C** — not 13 nodes, and emphatically not 1. That is the first case in the audit where the
split default and the static-kwarg exception both apply, to different parts of the *same*
switch, and it is a cleaner test of the open clause than any of the six recorded
deviations. **Flagged, not decided** — it is a policy call, per the policy doc.

## calls into other models

**None.** `run()` calls only `self.<staticmethod>`; nothing in the file touches another
`Model` instance. The reverse direction exists: `models/physics/confinement_time.py:203`
calls `PlasmaGeom.calculate_iter_physics_basis_elongation` as a bare staticmethod.

## JAX-difficulty flags

- **F1 — `if self.data.physics.kappa95 > 1.77:` (line 389), branch 10** —
  `needs-lax-cond-or-where`, severity `workaround-known`. A Python `if` on a freshly
  computed (therefore traced) value. Trivially `jnp.where`, **but see D6: the function it
  guards is not C¹**, so a `where` gives the right value and a discontinuous derivative.
  This is the only traced-value branch in the file.
- **F2 — `min(2.0e0, 1.5e0 + 0.5e0/(aspect - 1.0e0))` (lines 272, 285), branches 2 and 3** —
  `needs-lax-cond-or-where`, severity `minor` (`jnp.minimum`). Two notes: the clamp
  activates at `aspect == 2.0` (`0.5/(A-1) == 0.5`), a derivative kink; and `aspect`'s
  iteration-variable bounds are `(1.1, 10.0)` (ixc 1), so **the kink is strictly inside
  the optimiser's box**, as is the pole at `aspect == 1`.
- **F3 — `np.sqrt(...)` on a sign-unconstrained argument (line 380), branch 10** —
  severity `workaround-known`. The radicand is a quadratic in `aspect` plus
  `4a·m_s_limit`; the fit is documented valid only for `2.6 < aspect < 3.6`, and nothing
  enforces that. Outside it the radicand can go negative → `nan` with no error. Also a
  `safe_sqrt` site by the `density_limits.md` §Derivative-safe power laws argument.
- **F4 — fractional powers**, severity `minor`, all `safe_pow` candidates:
  `eps**2.8`, `eps**2.1` (branch 1), `(1.5/aspect)**0.4` (branch 9),
  `aspect**1.4` (branch 11), `(1.8/aspect)**0.4` (branch 12), and `kappa95**ratio`
  (branch 10, a *traced* exponent — `safe_pow` must handle that). Bases are all strictly
  positive on the physical domain, so the `x == 0` derivative poison is unlikely to be
  hit; flagged for completeness because the check is cheap and
  `Tier1Contract.test_gradient_finite_at_zero` will exercise it.
- **F5 — in-place mutation**: none. No arrays, no loops, no dynamic shapes anywhere in
  the file.
- **F6 — non-traceable external calls**: **none.** No CoolProp, no external library.
  `tokamak_scope.md`'s `coolprop` column should read clean for `i_plasma_geometry`.
- **F7 — `ProcessValueError` / `ValueError` on data-dependent conditions**, severity
  `workaround-known`: `output()` raises on `n_divertors ∉ {0,1,2}`, and
  `PlasmaGeometryModelType(...)` / `PlasmaShapeModelType(...)` raise `ValueError`
  off-range. All three are on *switch* values, not traced quantities, so they resolve at
  graph-assembly time and never reach a traced branch.
- **F8 — the domain failures in D1 below are the real traceability problem in this
  file**, not any of F1–F7: `plasma_angles_arcs` returns silently wrong-signed results
  over a reachable part of the input domain, and JAX will differentiate those just as
  happily as PROCESS evaluates them.

## suspected defects in PROCESS

Convention: **documented, not fixed.** Nothing in `process/` was touched. Each is marked
*confirmed* (measured in this env) or *unconfirmed* (read from source, not measured) —
`next_steps.md` §11.7 records several confident diagnoses that measurement overturned.

**D1 — `plasma_angles_arcs` has no branch selection, and returns negative geometry for
`kappa < 1 + triang`. Confirmed by measurement.**
Line 755: `denomo = (kappa**2 - n**2) / (2*n)` with `n = 1 + triang`; line 756:
`thetao = np.arctan(kappa / denomo)`. When `kappa < 1 + triang` the denominator is
negative, `arctan` returns the wrong branch (a negative half-angle instead of
`arctan(...) + pi`), and every downstream quantity silently flips sign. Measured at
`rminor = 2.6667`, `rmajor = 8.0`, `triang = 0.5`:

| `kappa` | `thetao` | perimeter | cross-section | volume | surface |
|---|---|---|---|---|---|
| 1.501 | +1.5701 | 21.155 | 32.417 | 1601.5 | 1039.6 |
| 1.499 | **−1.5701** | **−3.978** | **−17.816** | **−3273.2** | **−415.2** |
| 1.400 | −1.5019 | −3.154 | −13.267 | −3081.8 | −370.7 |

No exception, no warning — a negative plasma volume propagates into every downstream
model. Exactly at `kappa == 1 + triang`, `denomo == 0.0` and a plain
`ZeroDivisionError` is raised (`kappa = 1.5, triang = 0.5` reproduces it). Likewise
`triang == +1.0` (`t == 0`) and `triang == −1.0` (`n == 0`) both raise
`ZeroDivisionError`.

**Reachability is the sharp part.** `input.py` declares `kappa` range `(0.99, 5.0)` and
`triang` range `(-1.0, 1.0)`, and `InputVariable.range` is documented *"inclusive of the
endpoints"* — so `triang = 1.0` is a legal IN.DAT value that crashes, and
`kappa = 1.4, triang = 0.5` is a legal IN.DAT pair that yields a negative volume. Worse
for the port's own purposes: `ITERATION_VARIABLES[175]` gives `kappa` bounds `(0.0, 10.0)`
and `[174]` gives `triang` bounds `(0.0, 1.0)`, so **the harness's own fuzz domain
contains the whole failure region and both singularities.** Any fuzz sampling of
`plasma_angles_arcs` (or of `run()`) must constrain `kappa > 1 + triang`, and that
constraint should be written into the port as an explicit precondition rather than
inherited silently. The legacy `perim`/`xsect0` share the defect (same denominators,
verified algebraically identical).

**D2 — `f_vol_plasma` is silently ignored on the Sauter path. Confirmed by reading; not
exercised by any regression case.**
Lines 491–501 multiply the double-arc volume by `self.data.physics.f_vol_plasma`; line
476 assigns `sauter_geometry`'s volume directly, unmultiplied. `f_vol_plasma` is a
user-settable volume multiplier (default `1.0`, input range `0.001–10.0`, the renamed
`cvol`) that is never assigned by any model — a pure user knob. Setting it and then
selecting `i_plasma_shape = 1` or `i_plasma_current = 8` makes it a no-op with no
diagnostic. Whether that is intended (Sauter's volume is "already correct") or an
oversight is not decidable from the source; both readings are plausible, which is why
it is documented rather than normalised.

**D3 — `output()` can report the wrong shape model. Confirmed by reading.**
Line 467 selects Sauter geometry when `i_plasma_current == 8` **or** `i_plasma_shape == 1`.
Lines 563–567 print `PlasmaShapeModelType(self.data.physics.i_plasma_shape).full_name`
plus *"plasma shape model is used"* — reading `i_plasma_shape` alone. A run with
`i_plasma_current = 8, i_plasma_shape = 0` therefore computes Sauter geometry and prints
*"PROCESS Original Double Arc plasma shape model is used"*. Reporting-only, but it is
exactly the kind of thing that makes a regression-output diff untrustworthy.

**D4 — `q95_min` is produced by one geometry model but consumed by a constraint gated on
a different switch. Unconfirmed — read from source, no run performed.**
`.physics.q95_min` is assigned in exactly one place in `process/`
(`plasma_geometry.py:247`, `i_plasma_geometry == 1`). Its only consumer is
`core/solver/constraints.py:1212`, constraint 45 ("edge safety factor lower limit
(TART)"), which guards on `itart == 0` (raising if so) but **not** on
`i_plasma_geometry`. With `itart = 1` and any `i_plasma_geometry != 1`, `q95_min` keeps
its `DataStructure` default of `0.0` and constraint 45 degenerates to `q95 >= 0` —
satisfied by construction, silently. `low_aspect_ratio_DEMO.IN.DAT` and
`st_regression.IN.DAT` both use `i_plasma_geometry = 10`; whether either also activates
`icc = 45` was not checked, so this may already be live. Worth a five-minute check before
anyone treats it as hypothetical.

**D5 — `fvol` uses two different truncations of 2/3 within one expression. Confirmed by
measurement; dead code.**
Line 1163: `-0.66666666e0 * np.pi * zn**3` (the outboard term). Line 1174:
`-0.66666e0 * np.pi * zn**3` (the inboard term). The two differ in the 6th significant
figure, so the function is not just imprecise but *asymmetrically* imprecise between its
two halves. Measured against the live `plasma_volume` at three operating points, `fvol`
disagrees by 1.3e-3 to 2.8e-3 m³ (≈1.3e-6 relative) — consistent with the truncation, and
the sole reason the legacy/live pair does not agree to machine precision the way
`perim`/`xsect0` do. Dead in `process/`; still asserted against hardcoded expectations in
`tests/unit/models/physics/test_plasma_geom.py::test_fvol`.

**D6 — the `corner_fudge` branch (i_plasma_geometry = 10) is C⁰ but not C¹. Confirmed by
measurement.**
Lines 389–394. At `kappa95 = 1.77` the value is continuous (both arms give `1.77`), but
the one-sided derivatives are **1.0000 from below and 0.7290 from above** (central
differences at `1.77 ∓ 1e-7` and `1.77 ± 1e-7`, `h = 1e-7`). A finite-difference gradient
straddling the point returns something between the two, and VMCON sees a Jacobian that
depends on step size. `i_plasma_geometry = 10` is the model used by both
`low_aspect_ratio_DEMO.IN.DAT` and `st_regression.IN.DAT`, so this is a live path in the
tracked regression set. The `if` also has to become a `jnp.where` (F1), which will
faithfully reproduce the kink rather than remove it.

**D7 — four stale docstrings in `physics_variables.py`. Confirmed by cross-reading.**
The per-value lists have not been updated since geometry models 11 and 12 (Menard) were
added:

| field | docstring says "calculated if `i_plasma_geometry` =" | code actually calculates it for | missing |
|---|---|---|---|
| `kappa` | `1-5, 7 or 9-10` | 1,2,3,4,5,7,9,10,11,12 | 11, 12 |
| `kappa95` | `0-3, 6, or 8-10` | 0,1,2,3,6,8,9,10,11,12 | 11, 12 |
| `triang95` | `0-2, 6, 8 or 9` | 0,1,2,6,8,9,10,11,12 | 10, 11, 12 |
| `triang` | `1, 3-5 or 7` | 1,3,4,5,7 | — (correct) |

These matter more than usual here: they are the closest thing PROCESS has to a written
statement of this switch's ownership table, and three of the four are wrong. The enum
(§"the enum is a machine-readable ownership table") is the correct one and should be
preferred.

**D8 — two missing `else` arms. Unconfirmed as reachable; guarded by input validation.**
(a) `run()`'s 13 branches are 13 independent `if`s, not an `if/elif` chain with a
terminal `else: raise`. An `i_plasma_geometry` outside 0–12 therefore matches nothing,
leaves `kappa95`/`triang95` at their defaults, and computes a full geometry silently —
and only crashes much later in `output()`, where `PlasmaGeometryModelType(...)` raises
`ValueError`. `input.py`'s `range=(0, 12)` makes this unreachable through IN.DAT, so it
is a robustness gap rather than a live bug; noted because the *asymmetry* (silent in
`run`, fatal in `output`) is the thing worth not reproducing in the port.
(b) `output()`'s `itart` block (lines 539–554) is `if itart == 0 / elif itart == 1` with
no `else`, so any other value leaves `itart_r` unwritten and prints nothing — same
class, same guard.

**D9 — dead initialisers.** `xsi, xso, thetai, thetao, xi, xo = 0.0` (lines 216–221) are
all unconditionally overwritten before any read. Cosmetic; a Fortran-translation residue.

**D10 — `a_plasma_surface_outboard` is always the double-arc answer, even under Sauter.
Confirmed by reading; documented in-source as deliberate.**
Line 461 writes `xso` unconditionally, before the geometry-model split. The source
comment (lines 451–452) says so explicitly — *"These are not given by Sauter but the
outboard area is required by DCLL and divertor"*. The consequence is a real
inconsistency: under Sauter, `a_plasma_surface` (Sauter) and `a_plasma_surface_outboard`
(double-arc) come from different models and will not sum or ratio consistently, and
`models/blankets/dcll.py:820,838` consumes the mismatched pair. Also `xsi` is computed
and thrown away on that path, and — the part the comment does not cover — **the Sauter
path still evaluates `plasma_angles_arcs`, so it inherits all of D1's singularities**,
including for the negative-triangularity runs that are the entire reason
`i_plasma_current = 8` exists.

**D11 — the four legacy functions are dead and two are exact duplicates. Confirmed by
measurement.**
`surfa`, `perim`, `fvol`, `xsect0` have no caller in `process/`. Algebraically,
`perim`'s `denomi = (tri²+kap²-1)/(2(1-tri)) + tri` reduces to
`(kap² - (1-tri)²)/(2(1-tri))`, which is `plasma_angles_arcs`'s `denomi` exactly; the
same holds for `denomo`. Verified numerically at three operating points: `perim` vs
`plasma_poloidal_perimeter ∘ plasma_angles_arcs` and `xsect0` vs `plasma_cross_section ∘
plasma_angles_arcs` agree to 1–3 ULP; `surfa` vs `plasma_surface_area` agrees to ~1e-15
relative; `fvol` differs at ~1.3e-6 for D5's reason. **Recommend the port simply does not
carry them** — but they are still under unit test with hardcoded expectations, so
deleting them from `process/` is a separate, out-of-scope decision.

## open questions

1. **Eight nodes or thirteen?** §switches touched shows families A = {0,6,8},
   B = {4,5,7}, C = {11,12} have provably identical reads-sets, so
   `traceability_policy.md`'s static-kwarg exception applies *within* a family while the
   split default applies *across* families. That is the first case where both halves of
   the policy bind on one switch. Not decided here — it is the policy call the doc says
   should not be re-litigated per unit, and this file is a better test case for it than
   any of the six recorded deviations.
2. **What is the graph-assembly predicate for the Sauter arm, and who owns it?**
   `i_plasma_current == 8 or i_plasma_shape == 1` is evaluated in *this* file, but
   `i_plasma_current` is also `plasma_current.py`'s own topology switch. Either the
   assembler evaluates the disjunction once and hands both files the result, or the two
   files re-derive it independently and can drift. Flagging rather than choosing.
3. **Is the `kappa > 1 + triang` precondition (D1) enforceable, and by whom?** It is not
   a constraint in `numerics.icc`, not an input validation, and not an assertion. If the
   port declares it as a precondition, the harness's fuzz sampler must respect it (the
   `ITERATION_VARIABLES` bounds do not), and something in a real run must too — otherwise
   the port faithfully reproduces a negative plasma volume. This is the one item in this
   record that blocks writing a testable port, not just a tidy one.
4. **Should `a_plasma_surface_outboard` be one node or two?** Today it is unconditional
   and always double-arc (D10). Structurally it could be (a) one always-double-arc node,
   preserving PROCESS exactly, or (b) a per-arm node consistent with whichever
   `a_plasma_surface` was chosen, which changes numbers and breaks the regression
   tolerance. (a) is clearly what a faithful port does; recording (b) so nobody
   "improves" it later without noticing it is a behaviour change in DCLL.
5. **Does any tracked input actually run `icc = 45` with `i_plasma_geometry != 1`?**
   That is the measurement that turns D4 from unconfirmed to confirmed (or overturns it),
   and it is a grep plus one run. Deliberately not done here — this record is a source
   audit, and `next_steps.md` §11.7's history is the reason not to promote a confident
   reading to a finding without the measurement.
6. **Is the `PlasmaGeometryModelType` enum's four-tag table worth making load-bearing?**
   It is currently decorative — read only by `output()` for labels — but it is a correct,
   in-source, per-value statement of which of the four shape fields is an input and which
   is an output, and D7 shows the prose docstrings that duplicate it have already drifted.
   If graph assembly reads it directly, it stops being decorative and starts being
   checked. Not proposed as a change to `process/`; proposed as what the port's assembler
   should consume.
7. **Where does this record live?** See the note at the top — `schema.md`'s mirroring
   rule puts it under `units/models/physics/`, the task named `units/models/`.

## ported (2026-08-26)

Port: `functional_process/models/physics/plasma_geometry.py`. Tests:
`tests/functional_process/models/physics/test_plasma_geometry.py`. `50 passed, 50
skipped` on a plain run (gradient checks skip by default); `100 passed` with
`--fp-gradients`; `260 passed` with `--fp-gradients --fp-fuzz 5`.

**Scope: the minimal closure for the live configuration, plus the sibling
kappa95/triang95 computation.** `_audit/tokamak_boundary.md`'s `.tokamak.plasma_geom`
slot names five outputs (`a_plasma_poloidal`, `a_plasma_surface`, `eps`, `rminor`,
`vol_plasma`); this pass adds `kappa95`/`triang95` because they are produced by the same
`i_plasma_geometry` branch this pass already has to reason about (the "extraction seam"
and "proposed signature(s)" sections above already treat them as one unit).

Functions ported, matching the "proposed signature(s)" section above almost verbatim
(only `calculate_geometry_sauter`'s return order was chosen here, since that section
left it unstated):

| function | shape | matches proposed signature |
|---|---|---|
| `calculate_minor_radius(rmajor, aspect) -> (rminor, eps)` | new, unconditional | yes, exact |
| `plasma_angles_arcs(a, kappa, triang)` | verbatim port (function 3) | yes |
| `plasma_poloidal_perimeter(xi, thetai, xo, thetao)` | verbatim port (function 4) | yes |
| `plasma_surface_area(rmajor, rminor, xi, thetai, xo, thetao)` | verbatim port (function 5) | yes |
| `plasma_volume(rmajor, rminor, xi, thetai, xo, thetao)` | verbatim port (function 6) | yes |
| `plasma_cross_section(xi, thetai, xo, thetao)` | verbatim port (function 7) | yes |
| `sauter_geometry(a, r0, kappa, triang, square)` | verbatim port (function 8) | yes |
| `calculate_shape_ipdg89_x_point(kappa, triang) -> (kappa95, triang95)` | new, `i_plasma_geometry == 0` only | yes, exact |
| `calculate_geometry_double_arc(rmajor, rminor, kappa, triang, f_vol_plasma) -> (len_plasma_poloidal, vol_plasma, a_plasma_poloidal, a_plasma_surface)` | new, double-arc arm | yes, exact |
| `calculate_geometry_sauter(rmajor, rminor, kappa, triang, plasma_square) -> (len_plasma_poloidal, vol_plasma, a_plasma_poloidal, a_plasma_surface)` | new, Sauter arm, ported but unwired | reorders `sauter_geometry`'s own tuple for symmetry with the double-arc function; not specified by the proposed signature |

**Not ported in this pass** (each already discussed above, cross-referenced rather than
re-derived):

- `calculate_iter_physics_basis_elongation` (function 9) — **already ported**, by unit
  #10, `functional_process/models/physics/confinement_time.py` (its own module
  docstring records the cross-file dependency). Not duplicated here.
- The other 12 `i_plasma_geometry` values (1–12) — see "switches touched" above for each
  one's reads; none is live on any tracked regression input this pass checked.
- `i_plasma_wall_gap == 0` (writes `.build.dr_fw_plasma_gap_{inboard,outboard}`) — not
  live (`large_tokamak_eval` leaves it at the default `1`, which touches nothing in this
  file). The `==0` arm would need `.build` write access this unit hasn't claimed.
- The Sauter arm's occupant (compound switch `i_plasma_current == 8 or i_plasma_shape ==
  SAUTER`) — `calculate_geometry_sauter`/`sauter_geometry` are ported as functions (cheap,
  zero `self.data` access, per "the extraction seam" above) but not wired to a
  `PlasmaGeometryArm` occupant, since the arm is not live on any tracked input and (per
  "sample provenance is the weak point" above) has no regression oracle at all.
- `.physics.a_plasma_surface_outboard` (D10) — not a target output, consumed by
  `models/blankets/dcll.py`, outside this unit's scope.
- `PlasmaGeom.output()` and the four dead legacy functions (`surfa`, `perim`, `fvol`,
  `xsect0`) — reporting-only / dead-in-`process/` respectively (**D11**'s
  recommendation followed: not carried).

**Deviations from PROCESS: none beyond D1's faithful reproduction.**
`plasma_angles_arcs` is ported bit-for-bit, including the domain failure at
`kappa < 1 + triang` (D1) — not fixed, not guarded, matching every other faithfully-ported
defect in this project. Every sample and fuzz bound in the test file is chosen to avoid
that domain (documented in the test file's module docstring) rather than to test through
it, since D1 is a genuine PROCESS defect, not a porting question — testing "does the port
reproduce the wrong-signed geometry" is a separate, not-yet-written check, not part of
value/gradient agreement.

**Cottax nodes, resolving part of the open-questions list above:**

| class | family | owns | reads |
|---|---|---|---|
| `PlasmaMinorRadius` | (none — unconditional) | `.physics.rminor`, `.physics.eps` | `.physics.rmajor`, `.physics.aspect` |
| `Ipdg89XPointPlasmaShape` | `PlasmaShapeKappa95Triang95` | `.physics.kappa95`, `.physics.triang95` | `.physics.kappa`, `.physics.triang` |
| `DoubleArcPlasmaGeometry` | `PlasmaGeometryArm` | `.physics.len_plasma_poloidal`, `.physics.vol_plasma`, `.physics.a_plasma_poloidal`, `.physics.a_plasma_surface` | `.physics.rmajor`, `.physics.rminor`, `.physics.kappa`, `.physics.triang`, `.physics.f_vol_plasma` |

**Open question 1 (eight or thirteen?) is superseded, not answered.** The wave-1 binding
policy (`next_steps.md` §14.2, "no switch is a static kwarg") settles it in the
"thirteen" direction for any future pass that ports the other twelve values — no family
grouping by reads-identical sets, one occupant class per value ever supported. This pass
only adds the first of those thirteen (`Ipdg89XPointPlasmaShape`); the other twelve
remain to be ported the same way, individually, when a unit needs them.

**Open question 2 (Sauter predicate ownership) is now concrete, not resolved.**
`DoubleArcPlasmaGeometry` is the `False` arm of `i_plasma_current == 8 or i_plasma_shape
== SAUTER`; nothing currently instantiates the `True` arm. Whichever pass ports
`plasma_current.py`'s own `i_plasma_current` topology split needs to know this file's
`PlasmaGeometryArm` family exists and shares the same disjunction, so the two are wired
consistently rather than independently re-deriving it (per the audit record's own
phrasing above).

**Open questions 3, 4, 5, 6 are unchanged** — none was touched by this pass. Question 3
(the `kappa > 1 + triang` precondition) was worked around in the test file's sampling,
not resolved as a declared precondition anywhere a real run would enforce it; still
blocking for a fuzz sampler that does not already know to avoid it.
