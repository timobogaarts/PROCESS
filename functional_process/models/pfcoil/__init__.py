"""Pure-functional port of `process/models/pfcoil.py` (`PFCoil`, `CSCoil`).

A **package**, not a module, and deliberately so: `process/models/pfcoil.py` is 5286
lines and the largest wholly-unported model in the tokamak scope. The mirror-path rule
still holds -- the import path is `functional_process.cottax.pfcoil`, the same as a flat
`pfcoil.py` would give, and each submodule has its own record/test pair at the mirrored
path (`_audit/units/models/pfcoil/<stem>.md`,
`functional_process/tests/models/pfcoil/test_<stem>.py`).

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
  (read by `functional_process/cottax/structure.py::Structure`), and
- `.pf_coil.r_pf_coil_outer` (read by `functional_process/cottax/cryostat.py::Cryostat`).

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

**Since 2026-08-30 those constants are a `PFCoilTopology` instance**, not five loose
integers: the spherical tokamaks need a second one (`n_pf_coils_in_group = (2, 2, 2, 2)`
and no CS at all), and a package-wide module constant cannot describe two machines. The
old names are kept as aliases of `REFERENCE_TOPOLOGY` -- they are what every
conventional occupant and every existing harness case is written against.
"""

import dataclasses
import enum

NGC2 = 22
"""`pfcoil_variables.NGC2` -- the storage width of every per-coil array. Re-declared
here as a plain int so nothing in the package has to reach into `process` for a
shape."""


class PFLocation(enum.IntEnum):
    """`process/models/pfcoil.py:45`'s `PFLocationTypes`, re-declared as a plain enum.

    The four values `pfcoil()`'s placement loop (`:245-352`) dispatches on, and the same
    four `place_pf_*` helpers exist for. Re-declared rather than imported for the reason
    `NGC2` is: this package's shapes must not depend on importing `process`.
    """

    ABOVE_CS = 1
    ABOVE_TF = 2
    OUTSIDE_TF = 3
    GENERALLY_PLACED = 4


@dataclasses.dataclass(frozen=True)
class PFCoilTopology:
    """How many coils there are, in which groups, and whether there is a CS at all.

    **Graph-assembly data, not ports.** `n_pf_coil_groups`, `n_pf_coils_in_group`,
    `i_pf_location`, `iohcl` and `n_cs_current_filaments` between them fix every array
    index in this package -- which coil sits at which slot of the `NGC2`-wide arrays,
    where the CS sits if it exists, where the plasma sits, how many filaments the CS is
    split into. Per `_audit/naming_convention.md` § "Switches are not ports", a value
    read once to decide *which* subgraph exists is consumed by the code that assembles
    the `Graph`; this object is that value, gathered into one place instead of five
    module constants, so that a second machine can carry a second one.

    It is a **static** field of every node in the package (`eqx.field(static=True)`),
    the same shape `BootstrapCurrentFractionScaling` gives `n_plasma_profile_elements`
    and `PFCoilInductance` gives `NOH`: it changes the traced program, so it may not be
    a traced value, and a different topology is a different node instance rather than a
    different argument.

    Two instances exist: `REFERENCE_TOPOLOGY` (`large_tokamak_eval.IN.DAT`, and every
    conventional tokamak the port assembles) and `SPHERICAL_TOKAMAK_TOPOLOGY`
    (`spherical_tokamak_eval.IN.DAT`/`st_regression.IN.DAT`). `indat` chooses between
    them; nothing else constructs one.
    """

    n_pf_coil_groups: int
    """`.pf_coil.n_pf_coil_groups`."""

    n_pf_coils_in_group: tuple[int, ...]
    """`.pf_coil.n_pf_coils_in_group[:n_pf_coil_groups]`. PROCESS refuses more than
    `N_PF_COILS_IN_GROUP_MAX = 2` per group (`pfcoil.py:142-148`)."""

    i_pf_location: tuple[PFLocation, ...]
    """`.pf_coil.i_pf_location[:n_pf_coil_groups]` -- which `place_pf_*` each group
    takes."""

    has_central_solenoid: bool
    """`.build.iohcl != 0`. **False is absence, not a zero**: with no CS, `ohcalc` is
    never entered (`pfcoil.py:1048-1050`), no filament exists (`:202-204`), the CS's
    slot in every per-coil array does not exist either, and `.tokamak.cs_coil` is an
    empty namespace slot rather than a namespace of nodes computing zeros."""

    n_cs_current_filaments: int = 7
    """`.pf_coil.n_cs_current_filaments`, PROCESS default 7
    (`pfcoil_variables.py:315`). Filaments per half of the CS; ignored when
    `has_central_solenoid` is false, where PROCESS sets `nfxf = 0` outright."""

    def __post_init__(self):
        """Refuse a topology whose three per-group tuples disagree in length.

        Raises
        ------
        ValueError
            When `n_pf_coils_in_group` or `i_pf_location` has an entry count other
            than `n_pf_coil_groups`.
        """
        if len(self.n_pf_coils_in_group) != self.n_pf_coil_groups:
            raise ValueError(
                f"n_pf_coils_in_group has {len(self.n_pf_coils_in_group)} entries for "
                f"{self.n_pf_coil_groups} groups"
            )
        if len(self.i_pf_location) != self.n_pf_coil_groups:
            raise ValueError(
                f"i_pf_location has {len(self.i_pf_location)} entries for "
                f"{self.n_pf_coil_groups} groups"
            )

    @property
    def n_pf_coils(self) -> int:
        """PF coils, flattened over groups in `pfcoil()`'s group-then-coil order."""
        return sum(self.n_pf_coils_in_group)

    @property
    def cs_index(self) -> int:
        """The CS's slot in every `NGC2`-wide array -- `n_cs_pf_coils - 1` in PROCESS's
        spelling (`pfcoil.py:154`).

        Raises
        ------
        AttributeError
            When there is no central solenoid. Deliberately not `None`: an index that
            does not exist must not be usable as one, and a machine with `iohcl = 0`
            reaching this attribute is a port bug rather than a value question.
        """
        if not self.has_central_solenoid:
            raise AttributeError(
                "this machine has no central solenoid (iohcl = 0), so there is no "
                "cs_index -- see PFCoilTopology.has_central_solenoid"
            )
        return self.n_pf_coils

    @property
    def n_cs_pf_coils(self) -> int:
        """`.pf_coil.n_cs_pf_coils` -- the PF coils plus the CS if there is one."""
        return self.n_pf_coils + (1 if self.has_central_solenoid else 0)

    @property
    def plasma_index(self) -> int:
        """The plasma's slot, `n_pf_cs_plasma_circuits - 1` (`pfcoil.py:1067-1079`)."""
        return self.n_cs_pf_coils

    @property
    def nfxf(self) -> int:
        """`.pf_coil.nfxf` -- the CS split symmetrically into filaments
        (`pfcoil.py:202-206`); `0` when there is no CS.
        """
        return 2 * self.n_cs_current_filaments if self.has_central_solenoid else 0

    def coils_of_group(self, group: int) -> range:
        """The flattened coil indices belonging to one group."""
        start = sum(self.n_pf_coils_in_group[:group])
        return range(start, start + self.n_pf_coils_in_group[group])

    def first_coil_of_group(self, group: int) -> int:
        """The flattened index of a group's first coil -- the one
        `peak_b_field_at_pf_coil` is called for (`pfcoil.py:4611-4631`).
        """
        return sum(self.n_pf_coils_in_group[:group])

    def last_coil_of_group(self, group: int) -> int:
        """The flattened index of a group's last coil -- `induct`'s `ncoils - 1`
        (`pfcoil.py:1870-1871`).
        """
        return sum(self.n_pf_coils_in_group[: group + 1]) - 1

    def group_of_coil(self, coil: int) -> int:
        """Which group a flattened coil index belongs to.

        Raises
        ------
        IndexError
            If `coil` is not one of this topology's PF coils.
        """
        seen = 0
        for group, n in enumerate(self.n_pf_coils_in_group):
            seen += n
            if coil < seen:
                return group
        raise IndexError(coil)

    def groups_at(self, *locations: PFLocation) -> tuple[int, ...]:
        """The group indices whose `i_pf_location` is one of `locations`, in order."""
        return tuple(
            group
            for group, location in enumerate(self.i_pf_location)
            if location in locations
        )


REFERENCE_TOPOLOGY = PFCoilTopology(
    n_pf_coil_groups=4,
    n_pf_coils_in_group=(1, 1, 2, 2),
    i_pf_location=(
        PFLocation.ABOVE_TF,
        PFLocation.ABOVE_TF,
        PFLocation.OUTSIDE_TF,
        PFLocation.OUTSIDE_TF,
    ),
    has_central_solenoid=True,
)
"""`large_tokamak_eval.IN.DAT`'s topology (`:247-248`), and `large_tokamak_nof`'s and
`low_aspect_ratio_DEMO`'s. Six PF coils at 0-5, the CS at 6, the plasma at 7, 14 CS
filaments. Groups 0 and 1 hold one coil each (both `i_pf_location = 2`, above the TF,
one above and one below the midplane via `pfcoil()`'s `top_bottom` toggle); groups 2
and 3 hold a symmetric pair each, outside the TF."""

SPHERICAL_TOKAMAK_TOPOLOGY = PFCoilTopology(
    n_pf_coil_groups=4,
    n_pf_coils_in_group=(2, 2, 2, 2),
    i_pf_location=(
        PFLocation.ABOVE_TF,
        PFLocation.OUTSIDE_TF,
        PFLocation.OUTSIDE_TF,
        PFLocation.GENERALLY_PLACED,
    ),
    has_central_solenoid=False,
)
"""`spherical_tokamak_eval.IN.DAT`'s topology (`:69`, `:233`, `:237-238`) and
`st_regression.IN.DAT`'s (`:1485`, `:1755`, `:1788-1792`) -- byte-for-byte the same
four numbers on both files.

**Eight PF coils and no CS**, so the plasma sits at index 8 and there is no index the
CS owns. Group 0 is a top/bottom pair above the TF (both coils in *one* group, so
`pfcoil()`'s `top_bottom` toggle flips inside the group rather than between two
groups); groups 1 and 2 are symmetric pairs outside the TF; group 3 is
`i_pf_location = 4`, placed from `rref`/`zref` about the plasma centre -- the arm
`place_pf_generally` (`pfcoil.py:1345-1401`) exists for, which no conventional file
reaches."""

N_PF_GROUPS = REFERENCE_TOPOLOGY.n_pf_coil_groups
"""`.pf_coil.n_pf_coil_groups` on the reference run (`large_tokamak_eval.IN.DAT:248`).
An alias of `REFERENCE_TOPOLOGY`, kept because it is the shape every existing
harness case and every conventional occupant is written against."""

N_COILS_IN_GROUP = REFERENCE_TOPOLOGY.n_pf_coils_in_group
"""`.pf_coil.n_pf_coils_in_group[:4]` on the reference run (`:247`)."""

N_PF_COILS = REFERENCE_TOPOLOGY.n_pf_coils
"""Six PF coils, flattened over groups in `pfcoil()`'s own group-then-coil order."""

CS_INDEX = REFERENCE_TOPOLOGY.cs_index
"""Index 6: the Central Solenoid's slot in every `NGC2`-wide `.pf_coil.*` array.
`pfcoil()` spells this `n_cs_pf_coils - 1` (`pfcoil.py:154`, `iohcl != 0`)."""

N_CS_PF_COILS = REFERENCE_TOPOLOGY.n_cs_pf_coils
"""`.pf_coil.n_cs_pf_coils` = 7 -- six PF coils plus the CS."""

PLASMA_INDEX = REFERENCE_TOPOLOGY.plasma_index
"""Index 7: the plasma's slot, written at `pfcoil.py:1067-1079` ("Plasma size and
shape"). `n_pf_cs_plasma_circuits - 1` in PROCESS's spelling."""

N_CS_FILAMENTS = REFERENCE_TOPOLOGY.n_cs_current_filaments
"""`.pf_coil.n_cs_current_filaments`, PROCESS default 7 (`pfcoil_variables.py:315`),
not set in the reference file. Filaments per half of the CS."""

NFXF = REFERENCE_TOPOLOGY.nfxf
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
