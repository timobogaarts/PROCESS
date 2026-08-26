---
kind: model-unit
status: draft
confidence: medium
---

**Ported (partial by switch, full by function).** `pfcoil/currents.py` /
`test_currents.py`: `calculate_efc_currents` (with `fixb`, `mtrx`, `solv`, `rsid`),
`calculate_plasma_initiation_currents`, `calculate_equilibrium_currents`,
`calculate_cs_flux_swing`, `calculate_time_point_currents` — tier-1. Five cottax nodes:
`CSCurrentDensityPulseStart`, `PFCoilInitiationCurrents`, `PFCoilEquilibriumCurrents`,
`CSFluxSwing`, `PFCoilTimePointCurrents`.

## source

`process/models/pfcoil.py`:

| lines | what |
|---|---|
| `161-164` | `j_cs_pulse_start` |
| `366-405` | plasma-initiation ("flux swing coils") current solve |
| `456-598` | equilibrium current solve, `i_pf_current = 1`, conventional aspect ratio |
| `600-661` | CS flux swing → `f_j_cs_start_end_flat_top` |
| `663-728` | per-coil currents at the three time points |
| `1403-1506` | `PFCoil.efc` |
| `1567-1613` | `PFCoil.solv` (`@staticmethod`, `scipy.linalg.svd`) |
| `5063-5130` | `rsid` (`@numba.njit`) |
| `5133-5183` | `fixb` (`@numba.njit`) |
| `5186-5285` | `mtrx` (`@numba.njit`) |

UNPORTED:

| lines | what | why not |
|---|---|---|
| `411-454` | the `itart = 1, itartpf = 0` ST arm | a different occupant; it bypasses the SVD entirely and reads `aspect` where this arm reads coil heights |
| `462-483` | the `i_pf_location = 1` (`ABOVE_CS`) arm of the equilibrium branch | not reachable on this run |
| `681-685` | the `i_pf_current = 0` arm | currents read from `ccl0_ma`/`ccls_ma` as inputs instead of computed; a different occupant, and it inverts which of `ccl0`/`ccl0_ma` is owned |
| `658-661` | the `iohcl = 0` arm of the flux swing | forces the ratio to 1 and logs an error |
| `1615-1719` | `vsec` | excluded by the wave brief; it produces the volt-second *capability* of the PF set, which nothing in this closure reads |
| `1721-2020` | `induct` | **now ported** -- `inductance.py` / `inductance.md` |
| `1085-1113` | `c_pf_coil_turn` (per-coil current per turn, six time points) | reads `n_pf_cs_plasma_circuits` and the plasma row; feeds `pf_power`, not this pass's boundary |

## The cycle

**A genuine three-node SCC, and this is the finding the port exists to surface.**

```
PFCoilTimePointCurrents  --.pf_coil.c_pf_cs_coil_*_ma-->  PFCoilCurrentWaveform
        ^                                                          |
        |                                     .pf_coil.c_pf_cs_coils_peak_ma
.pf_coil.f_j_cs_start_end_flat_top                                 v
        |                                                    PFCoilSizes
   CSFluxSwing                                                     |
        ^                                          .pf_coil.n_pf_coil_turns
        |                                                          v
        +---.pf_coil.ind_pf_cs_plasma_mutual---  PFCoilInductance  <'
```

(`CSFluxSwing` also reads `.pf_coil.n_pf_coil_turns` directly, so the three-node loop
through `PFCoilSizes` is there too; the four-node one through `PFCoilInductance` is the
edge this pass added.)

PROCESS closes it by bootstrapping and brute force. `pfcoil.py:605-608`: on the first
visit it sets `ind_pf_cs_plasma_mutual[:, :] = 1.0` and `n_pf_coil_turns[:] = 100.0` and
clears `first_call`; `Caller.call_models` then re-runs the *entire* pipeline up to ten
times until the objective and constraints stop moving. That is Gauss-Seidel over an
undeclared SCC.

**Update: the cycle is four nodes, not three.** `inductance.py::PFCoilInductance`
ports `PFCoil.induct` and owns `.pf_coil.ind_pf_cs_plasma_mutual`, which `CSFluxSwing`
reads and which `PFCoilSizes` feeds. What was a boundary input is now an internal edge
of the block, and PROCESS's `first_call` seeding of the matrix is revealed as the
iteration's initial guess. See `inductance.md` § "The cycle, one node larger".

**This is Shape A, not Shape B** — checked against the wave brief's rule before
declaring it, and it does not dissolve. No node here reads a `VarPath` it owns:
`CSFluxSwing` reads `.pf_coil.n_pf_coil_turns`, whose real producer is `masses.py`'s
`PFCoilSizes`, a different node. So **no `FixedPointFunction` is used anywhere in this
package**. What the assembler gets is a four-node cycle for `Blocking` to find, needing
an explicit `Drive` (`FixedPoint` reproduces PROCESS's own iteration; `RootFind` on the
`n_pf_coil_turns` residual would converge faster). **That decision is not made here** —
see this unit's entry in the wave report.

`.pf_coil.ind_pf_cs_plasma_mutual` is **no longer** a boundary input. `CSFluxSwing`
reads one column of it, `[0:6, 7]`; the producer owns the array whole, for the reasons
`inductance.md` § "Whole-array ownership, with the evidence" sets out.

## data footprint

`CSCurrentDensityPulseStart`: reads `.pf_coil.j_cs_flat_top_end`,
`.pf_coil.f_j_cs_start_pulse_end_flat_top`; writes `.pf_coil.j_cs_pulse_start`
(`pfcoil.py:161-164`).

`PFCoilInitiationCurrents` (`:366-405`, plus `:206-234` for the filaments it builds
internally):

| VarPath | read/write | classification | note |
|---|---|---|---|
| `.physics.rmajor` | read | explicit-arg | `:378` |
| `.physics.rminor` | read | explicit-arg | `:377-378` |
| `.pf_coil.r_pf_coil_middle_group_array` | read | explicit-arg | `:399` |
| `.pf_coil.z_pf_coil_middle_group_array` | read | explicit-arg | `:400` |
| `.pf_coil.r_cs_middle` | read | explicit-arg | `:229` |
| `.pf_coil.dz_cs_full` | read | explicit-arg | `:230` |
| `.pf_coil.a_cs_poloidal` | read | explicit-arg | `:210` |
| `.pf_coil.j_cs_flat_top_end` | read | explicit-arg | `:210` |
| `.pf_coil.f_j_cs_start_pulse_end_flat_top` | read | explicit-arg | `:232` |
| `.pf_coil.alfapf` | read | explicit-arg | `:401` |
| `.pf_coil.ssq0` | write | explicit-arg | `:387` |
| `.pf_coil.ccl0` | write | explicit-arg | `:387` |
| `.pf_coil.j_cs_pulse_start` | read | **not declared** | `:366` guards the whole block; on this arm the CS is present and its current density is an iteration variable, never zero. Dropped branch, recorded here rather than declared as a read that only ever gates |

`PFCoilEquilibriumCurrents` (`:456-598`):

| VarPath | read/write | classification | note |
|---|---|---|---|
| `.physics.plasma_current` | read | explicit-arg | `:490`, `:565` |
| `.physics.kappa`, `.physics.rminor` | read | explicit-arg | `:494` |
| `.physics.rmajor` | read | explicit-arg | `:557`, `:566` |
| `.physics.aspect` | read | explicit-arg | `:568` |
| `.physics.beta_poloidal_vol_avg` | read | explicit-arg | `:569` |
| `.physics.ind_plasma_internal_norm` | read | explicit-arg | `:570` |
| `.pf_coil.r/z_pf_coil_middle_group_array` | read | explicit-arg | `:474-478`, `:543-554` |
| `.pf_coil.alfapf` | read | explicit-arg | `:591` |
| `.pf_coil.ccls` | write | explicit-arg | `:471`, `:489`, `:598` |
| `.physics.b_plasma_vertical_required` | write | explicit-arg | `:575` |

`CSFluxSwing` (`:600-657`): reads `.pf_coil.ccls`,
`.pf_coil.ind_pf_cs_plasma_mutual`, `.pf_coil.n_pf_coil_turns`,
`.physics.vs_plasma_ramp_required`, `.build.dr_cs_bore`, `.build.dr_cs`,
`.pf_coil.dz_cs_full`, `.pf_coil.a_cs_poloidal`, `.pf_coil.j_cs_flat_top_end`,
`.pf_coil.f_j_cs_start_pulse_end_flat_top`; writes
`.pf_coil.f_j_cs_start_end_flat_top` (`:648`).

`PFCoilTimePointCurrents` (`:663-728`): reads `.pf_coil.ccl0`, `.pf_coil.ccls`,
`.pf_coil.a_cs_poloidal`, `.pf_coil.j_cs_flat_top_end`,
`.pf_coil.f_j_cs_start_pulse_end_flat_top`, `.pf_coil.f_j_cs_start_end_flat_top`; writes
`.pf_coil.c_pf_cs_coil_pulse_start_ma`, `_flat_top_ma`, `_pulse_end_ma`,
`.pf_coil.ccl0_ma`, `.pf_coil.ccls_ma`.

## proposed signature(s)

```python
def calculate_efc_currents(rpts, zpts, brin, bzin, r_fix, z_fix, c_fix,
                           r_group, z_group, alfa, n_in_group) -> tuple  # ssq, ccls (10,)
def calculate_plasma_initiation_currents(...10 args...) -> tuple   # ssq0, ccl0 (10,)
def calculate_equilibrium_currents(...10 args...) -> tuple         # ccls (10,), b_vertical
def calculate_cs_flux_swing(...10 args...) -> float
def calculate_time_point_currents(...6 args...) -> tuple           # three (7,)
```

`n_in_group` is a Python tuple — a shape resolved at trace time, and `static_argnames`
in the contract. It is not a switch value being smuggled in as a kwarg: it carries no
model choice, only how many coils each group's row of the matrix sums over.

## cottax node

Five `ExplicitFunction`s in `functional_process/models/pfcoil/currents.py`; ownership as
above. `.pf_coil.r/z/c_pf_cs_current_filaments` are **owned by none of them** — see
`fields.md` § "Storage this package refuses to own".

## tier signal

**Tier 1.** `efc` is a single SVD solve of a fixed matrix, not an iteration: there is no
convergence criterion on either side, so PROCESS's answer *is* ground truth and a value
comparison is meaningful. (This is the distinction `test_harness.md` draws for tier 2 —
"an unchecked fixed-iteration loop" — and `solv` is not one.)

**Sample provenance.** Legacy points read off a converged in-process `SingleRun` of
`large_tokamak_eval.IN.DAT`.

**Only `calculate_efc_currents` has a contract of its own.** The other four are inline
stretches of `pfcoil()` with no separable PROCESS callable; their oracle is `pfcoil()`
itself, in `test_masses.py`'s `TestPFCoilChain`. Writing a hand-rolled "reference" would
have compared the port against a copy of the port.

**Value tolerance is `rtol = 5e-12`, not tier 1's `1e-12`,** for
`TestCalculateEfcCurrents`. Both sides form the same pseudo-inverse of the same matrix
but obtain the decomposition from different LAPACK driver calls (`scipy.linalg.svd`
against `jnp.linalg.svd`); measured `2.3e-13` relative on the reference point. The
justification is attached to the `Tolerance` object, not to a bare float.

## switches touched

| switch | reachable values | live on `large_tokamak_eval` | decision | evidence |
|---|---|---|---|---|
| `.pf_coil.i_pf_current` | `0`, `1` | `1` (default, `pfcoil_variables.py:279`) | **split** | `:408` gates the whole equilibrium solve; `:678-685` inverts which of `ccl0`/`ccl0_ma` is input and which is output |
| `.physics.itart` / `.physics.itartpf` | `0`/`1` | `0` / `0` | **split** | `:411` — the ST arm computes `ccls` from `aspect**1.6` and never calls `efc` |
| `.pf_coil.i_pf_location[i]` | `1`-`4` | `(2, 2, 3, 3)` | **split** | `:462-539` — `ABOVE_CS` zeroes the group current, `ABOVE_TF` fixes it analytically, `OUTSIDE_TF`/`GENERALLY_PLACED` make it an unknown. Three different read sets |
| `.build.iohcl` | `0`, `1` | `1` | **split** | `:202-204` (no filaments), `:626-661` (flux swing) |

**UNPORTED switch values** for `indat.py`'s `UNPORTED` table: `i_pf_current = 0`;
`itart = 1`; `iohcl = 0`; any `i_pf_location` pattern other than `(2, 2, 3, 3)`.

## calls into other models

None. `PFCoilInitiationCurrents` calls `geometry.place_cs_filaments`, which is this same
package.

## JAX-difficulty flags

- **`scipy.linalg.svd` → `jnp.linalg.svd`**, `full_matrices=False`. PROCESS's `solv`
  indexes only `umat[j, i]` for `i < n_pf_coil_groups <= 10`, which is inside the thin
  `U`. The pseudo-inverse is invariant to the per-column sign convention (`U` and `V`
  flip together), so the two decompositions need not agree columnwise for the answer to
  agree.
- **The decomposition is taken of `gmat[:, :n_groups]`, not of the padded matrix, and
  that is a gradient fix.** `mtrx` writes only the first `n_groups` of `gmat`'s ten
  columns, so the rest are structurally zero and contribute
  `N_PF_GROUPS_MAX - n_groups` *repeated zero* singular values. SVD's JVP divides by
  `sigma_i^2 - sigma_j^2`, so a repeated singular value makes the entire tangent `nan`
  the moment the perturbation direction reaches that block — which is exactly what
  `test_gradient_finite_at_zero` found at `alfa = 0`, the one input whose value is zero
  there while its tangent is not. Trimming the columns removes the degenerate block.
  The **values are unchanged**: for `A = [A_n | 0]`, `A`'s nonzero singular values,
  their left vectors and the leading block of their right vectors are exactly `A_n`'s,
  which is precisely what `solv`'s two sums index. This is the `safe_math.py` bargain —
  same number, finite derivative — applied to linear algebra rather than to `sqrt`.
- **`solv`'s carried `zvec`** (`:1605-1611`) — the running value is *not* reset when a
  singular value falls below `1e-10`, so that term silently reuses the previous `j`'s
  ratio. Reproduced by forward-filling the last admissible index with `jax.lax.cummax`,
  rather than "corrected" to zero. Generically inert (all used singular values are well
  above the floor) but it is behaviour, not an accident of formatting.
- **`np.fill_diagonal` on a slice** (`:5274-5277`) → an explicit per-group
  `gmat.at[2*npts + j, j].set(...)`.
- **`logger.warning` on `|f_j_cs_start_end_flat_top| > 1`** (`:652-657`) — pure
  reporting, dropped.
- No CoolProp, no `scipy.special`, no `scipy.optimize`.

## open questions

- **Which `Drive` should the three-node SCC get?** Reproducing PROCESS exactly means a
  `FixedPoint` seeded the way `first_call` seeds it. A `RootFind` on the
  `n_pf_coil_turns` residual would be the point of the rewrite. Needs the assembler's
  decision; flagged in the wave report.
- **`.pf_coil.ssq0` is owned but nothing reads it.** It is `efc`'s residual norm, kept
  because it is a real stored field and a free diagnostic of whether the initiation
  solve is any good. Same category as `structure.md`'s `fncmass`/`gsmass`.
- **`.pf_coil.ccl0_ma`/`.pf_coil.ccls_ma` change role with `i_pf_current`.** They are
  outputs on this arm and *inputs* on `i_pf_current = 0` (`:684-685`). A dual-role field
  across occupants is fine while only one occupant exists, but the second occupant
  cannot simply be another class binding the same `VarPath` in the other direction.
