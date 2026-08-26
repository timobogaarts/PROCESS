"""Pure-functional port of `process/models/pfcoil.py` (`PFCoil`, `CSCoil`).

A **package**, not a module, and deliberately so: `process/models/pfcoil.py` is 5286
lines and the largest wholly-unported model in the tokamak scope. The mirror-path rule
still holds -- the import path is `functional_process.models.pfcoil`, the same as a flat
`pfcoil.py` would give, and each submodule has its own record/test pair at the mirrored
path (`_audit/units/models/pfcoil/<stem>.md`,
`tests/functional_process/models/pfcoil/test_<stem>.py`).

The split follows PROCESS's own data flow inside `PFCoil.pfcoil()`, not an arbitrary
line count:

- `geometry.py`  -- CS geometry, CS filament placement, PF coil placement per group.
- `currents.py`  -- the SVD current solve (`efc`/`mtrx`/`fixb`/`solv`/`rsid`), the
                    plasma-initiation and equilibrium currents, the CS flux swing, and
                    the coil current waveforms.
- `fields.py`    -- the Green's-function field kernel (`calculate_b_field_at_point`)
                    and the per-group peak field at each PF coil's edges.
- `masses.py`    -- coil sizing (turns, cross-section, edges) and the conductor/steel
                    masses that this pass's boundary actually asks for.

**Scope.** This pass ports the minimal coherent closure of the *sizing* chain that
produces the three variables this wave's new consumers declare:

- `.pf_coil.m_pf_coil_conductor_total` and `.pf_coil.m_pf_coil_structure_total`
  (read by `functional_process/models/structure.py::Structure`), and
- `.pf_coil.r_pf_coil_outer` (read by `functional_process/models/cryostat.py::Cryostat`).

Everything reachable *only* through `induct()`, `vsec()`, `outpf()`, `outvolt()`,
`waveform`'s downstream `c_pf_coil_turn`, the CS stress/fatigue chain, and `superconpf`
is **UNPORTED** -- see each submodule's audit record for the per-item reason. The
superconductor critical-current fits are already ported in
`functional_process/models/physics/superconductors.py` and are *not* re-ported here;
this closure never needs them, because a coil's mass depends on its steel *area*, which
comes from the JxB force and the CS steel fraction, not from any critical current.

**The reference arm.** Every occupant class here answers exactly one switch cell, the
one live on `tests/regression/input_files/large_tokamak_eval.IN.DAT`:

| switch | value | source |
|---|---|---|
| `.build.iohcl` | `1` (CS present) | PROCESS default, `build_variables.py` |
| `.pf_coil.i_pf_conductor` | `0` = `SUPERCONDUCTING` | default, `pfcoil_vars:230` |
| `.pf_coil.i_pf_current` | `1` (currents computed, not input) | default, `:279` |
| `.pf_coil.i_pf_superconductor` | `3` | set in the IN.DAT |
| `.pf_coil.i_cs_superconductor` | `1` | set in the IN.DAT |
| `.pf_coil.i_pf_location` | `(2, 2, 3, 3)` | set in the IN.DAT |
| `.pf_coil.i_r_pf_outside_tf_placement` | `0` (follow the TF curve) | default, `:287`|
| `.tfcoil.i_tf_shape` | `1` = `D_SHAPE` | not `PICTURE_FRAME` |
| `.physics.itart` / `.physics.itartpf` | `0` / `0` | conventional aspect ratio |

**The coil-count topology is graph-assembly data, not a port.** `n_pf_coil_groups = 4`,
`n_pf_coils_in_group = (1, 1, 2, 2)`, `n_cs_current_filaments = 7` and `iohcl = 1`
between them fix every array index in this file: six PF coils at 0-5, the CS at 6, the
plasma at 7, `nfxf = 14` CS filaments. Per `_audit/naming_convention.md` § "Switches are
not ports", a value read once to decide *which* subgraph exists is consumed by the code
that assembles the `Graph`, not carried on a node -- so `n_cs_pf_coils`,
`n_pf_cs_plasma_circuits` and `nfxf`, which `pfcoil()` writes into `data` purely as loop
bookkeeping, are **not owned by any node here**. They are module constants below.
"""

N_PF_GROUPS = 4
"""`.pf_coil.n_pf_coil_groups` on the reference run (`large_tokamak_eval.IN.DAT:248`)."""

N_COILS_IN_GROUP = (1, 1, 2, 2)
"""`.pf_coil.n_pf_coils_in_group[:4]` on the reference run (`:247`). Group 0 and 1 hold
one coil each (both `i_pf_location = 2`, above the TF, one above and one below the
midplane via `pfcoil()`'s `top_bottom` toggle); groups 2 and 3 hold a symmetric pair
each (`i_pf_location = 3`, outside the TF)."""

N_PF_COILS = sum(N_COILS_IN_GROUP)
"""Six PF coils, flattened over groups in `pfcoil()`'s own group-then-coil order."""

CS_INDEX = N_PF_COILS
"""Index 6: the Central Solenoid's slot in every `NGC2`-wide `.pf_coil.*` array.
`pfcoil()` spells this `n_cs_pf_coils - 1` (`pfcoil.py:154`, `iohcl != 0`)."""

N_CS_PF_COILS = N_PF_COILS + 1
"""`.pf_coil.n_cs_pf_coils` = 7 -- six PF coils plus the CS."""

PLASMA_INDEX = N_CS_PF_COILS
"""Index 7: the plasma's slot, written at `pfcoil.py:1067-1079` ("Plasma size and
shape"). `n_pf_cs_plasma_circuits - 1` in PROCESS's spelling."""

NGC2 = 22
"""`pfcoil_variables.NGC2` -- the storage width of every per-coil array. Re-declared
here as a plain int so nothing in the package has to reach into `process` for a
shape."""

N_CS_FILAMENTS = 7
"""`.pf_coil.n_cs_current_filaments`, PROCESS default 7 (`pfcoil_variables.py:315`),
not set in the reference file. Filaments per half of the CS."""

NFXF = 2 * N_CS_FILAMENTS
"""`.pf_coil.nfxf` = 14 -- the CS split symmetrically into filaments
(`pfcoil.py:206`)."""

NPTS = 32
"""`npts`, the number of midplane test points for the plasma-initiation null-field
solve. A literal in `pfcoil()` (`:368`), bounded by `NPTSMX = 32`."""

N_PF_GROUPS_MAX = 10
"""`pfcoil_variables.N_PF_GROUPS_MAX`. The column width of the least-squares matrix,
carried verbatim because it changes the SVD's shape and therefore its answer."""

LROW1 = 2 * NPTS + N_PF_GROUPS_MAX
"""`lrow1 = 2 * NPTSMX + N_PF_GROUPS_MAX` = 74 (`pfcoil.py:105`), the row count of
`gmat`/`bvec`. Kept at the full padded height rather than trimmed to the rows actually
written, so that `mtrx`'s row indexing reads as PROCESS's does; the trailing all-zero
rows contribute nothing to the solve, and the trailing all-zero *columns* are trimmed
at the decomposition itself -- see `currents._solv` for why that one is not cosmetic."""
