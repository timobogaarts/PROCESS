"""Harness cases for `functional_process/models/pfcoil/geometry.py`.

Audit record: `functional_process/_audit/units/models/pfcoil/geometry.md`. Three tier-1
contracts, one per unit that PROCESS exposes as a callable of its own:

- `calculate_cs_geometry` -> `CSCoil.calculate_cs_geometry` (a `@staticmethod`, called
  directly; the adapter only unpacks its `CSGeometry` dataclass into the port's tuple).
- `place_cs_filaments` -> `CSCoil.place_cs_filaments` (likewise, plus the `[:NFXF]` trim
  the port documents).
- `calculate_pf_coil_group_positions` -> `PFCoil.place_pf_above_tf` and
  `PFCoil.place_pf_outside_tf`, driven in `pfcoil()`'s own group order with `pfcoil()`'s
  own `top_bottom` carry. Those two are *instance* methods and allocate their return
  arrays from `self.data.pf_coil.n_pf_coil_groups`, so the adapter binds a
  `DataStructure` carrying that one integer -- the only `self.data` read either makes,
  confirmed by reading both bodies (`process/models/pfcoil.py:1232-1240`, `:1303-1311`).

`calculate_pf_coil_positions` has **no contract here**, and that is deliberate: it ports
`pfcoil()`'s inline `ncl` flattening loop (`:663-672`) plus the CS's two array writes
(`:182`, `:186-188`), which PROCESS never exposes as a callable. It is covered instead by
`test_masses.py`'s whole-`pfcoil()` chain contract, where PROCESS's own `pfcoil()` is the
oracle. See `geometry.md` § tier signal.
"""

import numpy as np
from process.core.model import DataStructure
from process.models.pfcoil import PFCoil

from functional_process._harness import Tier1Contract, legacy_sample
from functional_process.models.pfcoil import N_PF_GROUPS, NFXF
from functional_process.models.pfcoil.geometry import (
    calculate_cs_geometry,
    calculate_pf_coil_group_positions,
    place_cs_filaments,
)


def _reference_cs_geometry(z_tf_inside_half, f_z_cs_tf_internal, dr_cs, dr_cs_bore):
    """`CSCoil.calculate_cs_geometry`, its dataclass unpacked in declaration order."""
    from process.models.pfcoil import CSCoil

    g = CSCoil.calculate_cs_geometry(
        z_tf_inside_half=z_tf_inside_half,
        f_z_cs_tf_internal=f_z_cs_tf_internal,
        dr_cs=dr_cs,
        dr_cs_bore=dr_cs_bore,
    )
    return (
        g.z_cs_coil_upper,
        g.z_cs_coil_lower,
        g.r_cs_coil_middle,
        g.r_cs_middle,
        g.z_cs_coil_middle,
        g.r_cs_coil_outer,
        g.r_cs_coil_inner,
        g.a_cs_poloidal,
        g.a_cs_toroidal,
        g.dz_cs_full,
        g.dr_cs_full,
    )


def _reference_place_cs_filaments(
    r_cs_middle, z_cs_inside_half, c_cs_flat_top_end, f_j_cs_start_pulse_end_flat_top
):
    """`CSCoil.place_cs_filaments`, trimmed from `NFIXMX` to the `NFXF` filled entries.

    `n_cs_current_filaments` and `nfxf` are the port's module constants, not arguments
    (see `geometry.place_cs_filaments`); they are passed here as the same literals so
    the two sides describe the same coil.
    """
    from process.models.pfcoil import CSCoil

    r, z, c = CSCoil.place_cs_filaments(
        n_cs_current_filaments=NFXF // 2,
        r_cs_middle=r_cs_middle,
        z_cs_inside_half=z_cs_inside_half,
        c_cs_flat_top_end=c_cs_flat_top_end,
        f_j_cs_start_pulse_end_flat_top=f_j_cs_start_pulse_end_flat_top,
        nfxf=NFXF,
    )
    return r[:NFXF], z[:NFXF], c[:NFXF]


def _reference_pf_coil_group_positions(
    rmajor,
    rminor,
    triang,
    rpf2,
    z_tf_top,
    dz_tf_upper_lower_midplane,
    zref,
    r_pf_outside_tf_midplane,
):
    """`pfcoil()`'s placement loop for `i_pf_location = (2, 2, 3, 3)`.

    Runs PROCESS's own `place_pf_above_tf` / `place_pf_outside_tf` in group order,
    carrying `top_bottom` between calls exactly as `pfcoil()` does
    (`process/models/pfcoil.py:127`, `:272-300`, `:302-323`). Both methods write only the
    one group's row of the array they return, so the rows are gathered here.
    """
    data = DataStructure()
    data.pf_coil.n_pf_coil_groups = N_PF_GROUPS
    model = PFCoil(cs_fatigue=None, cs_coil=None)
    model.data = data

    n_pf_coils_in_group = np.array([1, 1, 2, 2], dtype=int)
    zref_full = np.concatenate([np.asarray(zref, dtype=float), np.ones(6)])

    r_group = np.zeros((N_PF_GROUPS, 2))
    z_group = np.zeros((N_PF_GROUPS, 2))

    top_bottom = 1
    for group in (0, 1):
        r, z, top_bottom = model.place_pf_above_tf(
            n_pf_coils_in_group=n_pf_coils_in_group,
            n_pf_group=group,
            rmajor=rmajor,
            triang=triang,
            rminor=rminor,
            itart=0,
            itartpf=0,
            z_tf_inside_half=0.0,
            dz_tf_upper_lower_midplane=dz_tf_upper_lower_midplane,
            z_tf_top=z_tf_top,
            top_bottom=top_bottom,
            rpf2=rpf2,
            zref=zref_full,
        )
        r_group[group] = r[group]
        z_group[group] = z[group]

    for group in (2, 3):
        r, z = model.place_pf_outside_tf(
            n_pf_coils_in_group=n_pf_coils_in_group,
            n_pf_group=group,
            rminor=rminor,
            zref=zref_full,
            i_tf_shape=1,
            i_r_pf_outside_tf_placement=0,
            r_pf_outside_tf_midplane=r_pf_outside_tf_midplane,
        )
        r_group[group] = r[group]
        z_group[group] = z[group]

    return r_group, z_group


class TestCalculateCsGeometry(Tier1Contract):
    """`calculate_cs_geometry` -> `CSCoil.calculate_cs_geometry`."""

    audit_record = "models/pfcoil/geometry.md"
    reference = _reference_cs_geometry
    ported = calculate_cs_geometry

    # Read off a converged PROCESS run of `large_tokamak_eval.IN.DAT`, in-process.
    samples = [
        legacy_sample(
            "large-tokamak-converged",
            z_tf_inside_half=8.818217164127494,
            f_z_cs_tf_internal=0.9,
            dr_cs=0.546816593988753,
            dr_cs_bore=2.003843190236783,
        ),
    ]

    fuzz_bounds = {
        "z_tf_inside_half": (2.0, 20.0),
        "f_z_cs_tf_internal": (0.5, 1.0),
        "dr_cs": (0.05, 2.0),
        "dr_cs_bore": (0.2, 6.0),
    }


class TestPlaceCsFilaments(Tier1Contract):
    """`place_cs_filaments` -> `CSCoil.place_cs_filaments`."""

    audit_record = "models/pfcoil/geometry.md"
    reference = _reference_place_cs_filaments
    ported = place_cs_filaments

    samples = [
        legacy_sample(
            "large-tokamak-converged",
            r_cs_middle=2.2772514872311596,
            z_cs_inside_half=15.87279089542949 / 2,
            # `-(a_cs_poloidal * j_cs_flat_top_end)`, `pfcoil.py:209-211`.
            c_cs_flat_top_end=-(8.679505454534445 * 21443595.371072624),
            f_j_cs_start_pulse_end_flat_top=0.93491189654662,
        ),
    ]

    fuzz_bounds = {
        "r_cs_middle": (0.5, 8.0),
        "z_cs_inside_half": (1.0, 12.0),
        "c_cs_flat_top_end": (-4.0e8, -1.0e7),
        "f_j_cs_start_pulse_end_flat_top": (0.5, 1.2),
    }


class TestCalculatePFCoilGroupPositions(Tier1Contract):
    """`calculate_pf_coil_group_positions` -> `place_pf_above_tf`/`place_pf_outside_tf`."""

    audit_record = "models/pfcoil/geometry.md"
    reference = _reference_pf_coil_group_positions
    ported = calculate_pf_coil_group_positions

    samples = [
        legacy_sample(
            "large-tokamak-converged",
            rmajor=8.0,
            rminor=2.6666666666666665,
            triang=0.5,
            rpf2=-1.825,
            z_tf_top=8.784333333333333,
            dz_tf_upper_lower_midplane=-1.233883830794161,
            zref=np.array([3.6, 1.2, 1.0, 2.8]),
            r_pf_outside_tf_midplane=17.078406000060053,
        ),
    ]

    fuzz_bounds = {
        "rmajor": (4.0, 14.0),
        "rminor": (1.0, 4.0),
        "triang": (0.1, 0.7),
        "rpf2": (-3.0, -0.5),
        "z_tf_top": (4.0, 14.0),
        "dz_tf_upper_lower_midplane": (-3.0, 3.0),
        # `zref` must keep `rminor * zref < r_pf_outside_tf_midplane` for the D-shaped
        # radius `sqrt(r^2 - z^2)`; the bounds above make the worst case
        # `4.0 * 3.0 = 12.0 < 18.0`.
        "zref": (np.full(4, 0.5), np.full(4, 3.0)),
        "r_pf_outside_tf_midplane": (18.0, 26.0),
    }
