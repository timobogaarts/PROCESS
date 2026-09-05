"""Harness cases for `functional_process/cottax/pfcoil/fields.py`.

Audit record: `functional_process/_audit/units/models/pfcoil/fields.md`. Three tier-1
contracts:

- `calculate_b_field_at_point` -> PROCESS's `@numba.njit` kernel of the same name, called
  directly with the same signature. No adapter at all: it takes no `self` and no `data`.
- `calculate_coil_current_waveform` -> `PFCoil.waveform`, which writes into `data`; the
  adapter binds a `DataStructure` carrying the one integer it reads
  (`n_cs_pf_coils`) and reads the two arrays back.
- `calculate_pf_coil_peak_fields` -> `PFCoil.waveform` followed by four calls to
  `peak_b_field_at_pf_coil`, driven in `pfcoil()`'s own group order
  (`process/models/pfcoil.py:854-866`).

**The third adapter reproduces one piece of `pfcoil()`'s state by re-running PROCESS's
own code, not by asserting it.** `peak_b_field_at_pf_coil` overwrites the *currents* of
the `nfxf` CS filaments but reads their *positions* from whatever
`.pf_coil.r/z_pf_cs_current_filaments` already hold; `pfcoil()` leaves entries 0 and 1
holding the group-0 and group-1 PF coil positions, because its equilibrium branch
overwrote them (`:474-479`). The adapter therefore calls `CSCoil.place_cs_filaments` --
PROCESS's function, not a copy of it -- and then applies that same two-entry overwrite,
so the oracle sees exactly the state PROCESS's own routine sees. See `fields.md`
§ "A PROCESS defect ported faithfully".
"""

import numpy as np

from functional_process.cottax._harness import Tier1Contract, legacy_sample
from functional_process.cottax.pfcoil import (
    N_COILS_IN_GROUP,
    N_CS_FILAMENTS,
    N_CS_PF_COILS,
    N_PF_COILS,
    N_PF_GROUPS,
    NFXF,
    NGC2,
)
from functional_process.cottax.pfcoil.fields import (
    calculate_b_field_at_point,
    calculate_coil_current_waveform,
    calculate_cs_bore_magnetic_field,
    calculate_cs_peak_fields,
    calculate_cs_self_peak_magnetic_field,
    calculate_pf_coil_peak_fields,
)
from process.core.model import DataStructure
from process.models.pfcoil import (
    CSCoil,
    PFCoil,
    peak_b_field_at_pf_coil,
)
from process.models.pfcoil import (
    calculate_b_field_at_point as process_calculate_b_field_at_point,
)


def _around(base, fraction):
    """Elementwise `(lower, upper)` at `+-fraction` of `base`, sign-safe.

    Fuzzing a coil geometry component-independently around the converged point rather
    than over an invented absolute range: the peak-field calculation has no domain guard
    to trip, but a filament set drawn from unrelated decades tells you nothing about
    whether the port sums PROCESS's terms in PROCESS's order.
    """
    base = np.asarray(base, dtype=float)
    low = base * (1.0 - fraction)
    high = base * (1.0 + fraction)
    return np.minimum(low, high), np.maximum(low, high)


# Every literal below is read off a converged, in-process PROCESS run of
# `tests/regression/input_files/large_tokamak_eval.IN.DAT`.
_C_START = np.array([
    15.205303476903685,
    19.82875078205095,
    0.5691387224018238,
    0.5691387224018238,
    0.07938149922724254,
    0.07938149922724254,
    174.00561799644518,
])
_C_FLAT = np.array([
    18.55559309814437,
    21.285004003992032,
    -6.9588680122398365,
    -6.9588680122398365,
    -4.8297169904629715,
    -4.8297169904629715,
    32.448166193521075,
])
_C_END = np.array([
    -0.5437459921847787,
    -3.6218342909013628,
    -7.6737615666565935,
    -7.6737615666565935,
    -4.929427869206218,
    -4.929427869206218,
    -186.11980298805437,
])
_R_MID = np.array([
    5.566666666666666,
    5.566666666666666,
    16.868931216641325,
    16.868931216641325,
    15.359714853856374,
    15.359714853856374,
    2.2772514872311596,
])
_Z_MID = np.array([
    9.644333333333332,
    -10.878217164127493,
    2.6666666666666665,
    -2.6666666666666665,
    7.466666666666666,
    -7.466666666666666,
    0.0,
])
_R_IN = np.array([
    4.917268464514993,
    4.871145214086073,
    16.303475590374013,
    16.303475590374013,
    14.967229668259906,
    14.967229668259906,
    2.003843190236783,
])
_R_OUT = np.array([
    6.21606486881834,
    6.26218811924726,
    17.434386842908637,
    17.434386842908637,
    15.752200039452843,
    15.752200039452843,
    2.550659784225536,
])
_Z_UP = np.array([
    10.293731535485007,
    -11.573738616708086,
    3.232122292933979,
    -3.232122292933979,
    7.8591518522631345,
    -7.8591518522631345,
    7.936395447714745,
])
_Z_LO = np.array([
    8.994935131181657,
    -10.1826957115469,
    2.1012110403993542,
    -2.1012110403993542,
    7.074181481070197,
    -7.074181481070197,
    -7.936395447714745,
])
_R_GROUP = np.array([
    [5.566666666666666, 0.0],
    [5.566666666666666, 0.0],
    [16.868931216641325, 16.868931216641325],
    [15.359714853856374, 15.359714853856374],
])
_Z_GROUP = np.array([
    [9.644333333333332, 0.0],
    [-10.878217164127493, 0.0],
    [2.6666666666666665, -2.6666666666666665],
    [7.466666666666666, -7.466666666666666],
])
_R_CS_MIDDLE = 2.2772514872311596
_DZ_CS_FULL = 15.87279089542949
_A_CS_POLOIDAL = 8.679505454534445
_J_CS_PULSE_START = 20047872.417147826
_J_CS_FLAT_TOP_END = 21443595.371072624
_RMAJOR = 8.0
_PLASMA_CURRENT = 16091095.408042267


def _reference_b_field_at_point(
    r_current_loop, z_current_loop, c_current_loop, r_test_point, z_test_point
):
    """PROCESS's kernel, called directly -- no `self`, no `data`, nothing to adapt."""
    return process_calculate_b_field_at_point(
        r_current_loop=np.asarray(r_current_loop, dtype=float),
        z_current_loop=np.asarray(z_current_loop, dtype=float),
        c_current_loop=np.asarray(c_current_loop, dtype=float),
        r_test_point=float(r_test_point),
        z_test_point=float(z_test_point),
    )


def _reference_coil_current_waveform(
    c_pf_cs_coil_pulse_start_ma, c_pf_cs_coil_flat_top_ma, c_pf_cs_coil_pulse_end_ma
):
    """`PFCoil.waveform`, restricted to the seven real coils.

    `waveform` reads only `.pf_coil.n_cs_pf_coils` and the three current arrays, and
    writes only the two arrays read back here (`process/models/pfcoil.py:2869-2940`).
    """
    data = DataStructure()
    data.pf_coil.n_cs_pf_coils = N_CS_PF_COILS
    data.pf_coil.c_pf_cs_coil_pulse_start_ma = np.zeros(NGC2)
    data.pf_coil.c_pf_cs_coil_flat_top_ma = np.zeros(NGC2)
    data.pf_coil.c_pf_cs_coil_pulse_end_ma = np.zeros(NGC2)
    data.pf_coil.c_pf_cs_coil_pulse_start_ma[:N_CS_PF_COILS] = (
        c_pf_cs_coil_pulse_start_ma
    )
    data.pf_coil.c_pf_cs_coil_flat_top_ma[:N_CS_PF_COILS] = c_pf_cs_coil_flat_top_ma
    data.pf_coil.c_pf_cs_coil_pulse_end_ma[:N_CS_PF_COILS] = c_pf_cs_coil_pulse_end_ma

    model = PFCoil(cs_fatigue=None, cs_coil=None)
    model.data = data
    model.waveform()
    return (
        data.pf_coil.c_pf_cs_coils_peak_ma[:N_CS_PF_COILS],
        data.pf_coil.f_c_pf_cs_peak_time_array[:N_CS_PF_COILS],
    )


def _reference_pf_coil_peak_fields(
    c_pf_cs_coil_pulse_start_ma,
    c_pf_cs_coil_flat_top_ma,
    c_pf_cs_coil_pulse_end_ma,
    r_pf_coil_middle,
    z_pf_coil_middle,
    r_pf_coil_inner,
    r_pf_coil_outer,
    z_pf_coil_upper,
    z_pf_coil_lower,
    r_pf_coil_middle_group_array,
    z_pf_coil_middle_group_array,
    r_cs_middle,
    dz_cs_full,
    a_cs_poloidal,
    j_cs_pulse_start,
    j_cs_flat_top_end,
    rmajor,
    plasma_current,
):
    """`PFCoil.waveform` then `peak_b_field_at_pf_coil`, once per group.

    See this module's docstring for why the CS filament positions are built by calling
    `CSCoil.place_cs_filaments` and then applying `pfcoil()`'s own two-entry overwrite.
    """
    data = DataStructure()
    p = data.pf_coil
    data.build.iohcl = 1
    p.n_cs_pf_coils = N_CS_PF_COILS
    p.n_pf_coil_groups = N_PF_GROUPS
    p.n_pf_coils_in_group = np.array([*N_COILS_IN_GROUP, 1, 0, 0, 0, 0, 0, 0, 0])
    p.n_cs_current_filaments = N_CS_FILAMENTS
    p.nfxf = NFXF

    for name, value in (
        ("c_pf_cs_coil_pulse_start_ma", c_pf_cs_coil_pulse_start_ma),
        ("c_pf_cs_coil_flat_top_ma", c_pf_cs_coil_flat_top_ma),
        ("c_pf_cs_coil_pulse_end_ma", c_pf_cs_coil_pulse_end_ma),
        ("r_pf_coil_middle", r_pf_coil_middle),
        ("z_pf_coil_middle", z_pf_coil_middle),
        ("r_pf_coil_inner", r_pf_coil_inner),
        ("r_pf_coil_outer", r_pf_coil_outer),
        ("z_pf_coil_upper", z_pf_coil_upper),
        ("z_pf_coil_lower", z_pf_coil_lower),
    ):
        full = np.zeros(NGC2)
        full[:N_CS_PF_COILS] = np.asarray(value, dtype=float)
        setattr(p, name, full)

    p.a_cs_poloidal = a_cs_poloidal
    p.j_cs_pulse_start = j_cs_pulse_start
    p.j_cs_flat_top_end = j_cs_flat_top_end
    p.b_pf_coil_peak = np.zeros(NGC2)
    p.bpf2 = np.zeros(NGC2)
    data.physics.rmajor = rmajor
    data.physics.plasma_current = plasma_current

    r_fil, z_fil, c_fil = CSCoil.place_cs_filaments(
        n_cs_current_filaments=N_CS_FILAMENTS,
        r_cs_middle=r_cs_middle,
        z_cs_inside_half=dz_cs_full / 2.0,
        c_cs_flat_top_end=0.0,
        f_j_cs_start_pulse_end_flat_top=0.0,
        nfxf=NFXF,
    )
    # `pfcoil.py:474-479`: the equilibrium branch overwrites filaments 0 and 1 with the
    # first coil of groups 0 and 1, and nothing restores them.
    for slot in (0, 1):
        r_fil[slot] = r_pf_coil_middle_group_array[slot, 0]
        z_fil[slot] = z_pf_coil_middle_group_array[slot, 0]
    p.r_pf_cs_current_filaments = r_fil
    p.z_pf_cs_current_filaments = z_fil
    p.c_pf_cs_current_filaments = c_fil

    model = PFCoil(cs_fatigue=None, cs_coil=None)
    model.data = data
    model.waveform()

    first_coil = 0
    for group in range(N_PF_GROUPS):
        peak_b_field_at_pf_coil(
            n_coil=first_coil + 1,
            n_coil_group=group + 1,
            t_b_field_peak=0,
            data=data,
        )
        first_coil += N_COILS_IN_GROUP[group]

    return p.b_pf_coil_peak[:N_PF_COILS], p.bpf2[:N_PF_COILS]


class TestCalculateBFieldAtPoint(Tier1Contract):
    """`calculate_b_field_at_point` -> the `@numba.njit` kernel it ports."""

    audit_record = "models/pfcoil/fields.md"
    reference = _reference_b_field_at_point
    ported = calculate_b_field_at_point

    # `ohcalc`'s end-of-flat-top call: the six PF coils plus the plasma, evaluated at
    # the CS's inner edge. Captured from a live traced run of `pfcoil()` on
    # `large_tokamak_eval.IN.DAT`.
    samples = [
        legacy_sample(
            "large-tokamak-converged",
            r_current_loop=np.array([
                5.566666666666666,
                5.566666666666666,
                16.868931216641325,
                16.868931216641325,
                15.359714853856374,
                15.359714853856374,
                8.0,
            ]),
            z_current_loop=np.array([
                9.644333333333332,
                -10.878217164127493,
                2.6666666666666665,
                -2.6666666666666665,
                7.466666666666666,
                -7.466666666666666,
                0.0,
            ]),
            c_current_loop=np.array([
                -543745.9921847787,
                -3621834.290901363,
                -7673761.566656593,
                -7673761.566656593,
                -4929427.869206218,
                -4929427.869206218,
                16091095.408042267,
            ]),
            r_test_point=2.003843190236783,
            z_test_point=0.0,
        ),
    ]

    fuzz_bounds = {
        "r_current_loop": (np.full(5, 1.0), np.full(5, 20.0)),
        "z_current_loop": (np.full(5, -12.0), np.full(5, 12.0)),
        "c_current_loop": (np.full(5, -2.0e7), np.full(5, 2.0e7)),
        "r_test_point": (0.5, 20.0),
        "z_test_point": (-12.0, 12.0),
    }


class TestCalculateCoilCurrentWaveform(Tier1Contract):
    """`calculate_coil_current_waveform` -> `PFCoil.waveform`."""

    audit_record = "models/pfcoil/fields.md"
    reference = _reference_coil_current_waveform
    ported = calculate_coil_current_waveform

    samples = [
        legacy_sample(
            "large-tokamak-converged",
            c_pf_cs_coil_pulse_start_ma=_C_START,
            c_pf_cs_coil_flat_top_ma=_C_FLAT,
            c_pf_cs_coil_pulse_end_ma=_C_END,
        ),
    ]

    # Opposite signs across the three time points, so the peak-selection branches are
    # exercised rather than always landing on the same one; magnitudes kept away from
    # zero so the normalisation never divides by a peak of zero.
    fuzz_bounds = {
        "c_pf_cs_coil_pulse_start_ma": (np.full(7, 0.5), np.full(7, 60.0)),
        "c_pf_cs_coil_flat_top_ma": (np.full(7, -60.0), np.full(7, -0.5)),
        "c_pf_cs_coil_pulse_end_ma": (np.full(7, -80.0), np.full(7, -0.5)),
    }


class TestCalculatePFCoilPeakFields(Tier1Contract):
    """`calculate_pf_coil_peak_fields` -> `waveform` + four `peak_b_field_at_pf_coil`."""

    audit_record = "models/pfcoil/fields.md"
    reference = _reference_pf_coil_peak_fields
    ported = calculate_pf_coil_peak_fields

    samples = [
        legacy_sample(
            "large-tokamak-converged",
            c_pf_cs_coil_pulse_start_ma=_C_START,
            c_pf_cs_coil_flat_top_ma=_C_FLAT,
            c_pf_cs_coil_pulse_end_ma=_C_END,
            r_pf_coil_middle=_R_MID,
            z_pf_coil_middle=_Z_MID,
            r_pf_coil_inner=_R_IN,
            r_pf_coil_outer=_R_OUT,
            z_pf_coil_upper=_Z_UP,
            z_pf_coil_lower=_Z_LO,
            r_pf_coil_middle_group_array=_R_GROUP,
            z_pf_coil_middle_group_array=_Z_GROUP,
            r_cs_middle=_R_CS_MIDDLE,
            dz_cs_full=_DZ_CS_FULL,
            a_cs_poloidal=_A_CS_POLOIDAL,
            j_cs_pulse_start=_J_CS_PULSE_START,
            j_cs_flat_top_end=_J_CS_FLAT_TOP_END,
            rmajor=_RMAJOR,
            plasma_current=_PLASMA_CURRENT,
        ),
    ]

    fuzz_bounds = {
        "c_pf_cs_coil_pulse_start_ma": _around(_C_START, 0.15),
        "c_pf_cs_coil_flat_top_ma": _around(_C_FLAT, 0.15),
        "c_pf_cs_coil_pulse_end_ma": _around(_C_END, 0.15),
        "r_pf_coil_middle": _around(_R_MID, 0.10),
        "z_pf_coil_middle": _around(_Z_MID, 0.10),
        "r_pf_coil_inner": _around(_R_IN, 0.05),
        "r_pf_coil_outer": _around(_R_OUT, 0.05),
        "z_pf_coil_upper": _around(_Z_UP, 0.05),
        "z_pf_coil_lower": _around(_Z_LO, 0.05),
        "r_pf_coil_middle_group_array": _around(_R_GROUP, 0.10),
        "z_pf_coil_middle_group_array": _around(_Z_GROUP, 0.10),
        "r_cs_middle": _around(_R_CS_MIDDLE, 0.15),
        "dz_cs_full": _around(_DZ_CS_FULL, 0.15),
        "a_cs_poloidal": _around(_A_CS_POLOIDAL, 0.15),
        "j_cs_pulse_start": _around(_J_CS_PULSE_START, 0.15),
        "j_cs_flat_top_end": _around(_J_CS_FLAT_TOP_END, 0.15),
        "rmajor": _around(_RMAJOR, 0.10),
        "plasma_current": _around(_PLASMA_CURRENT, 0.15),
    }


# ======================================================== the CS's own peak field
#
# `ohcalc`'s field block, added 2026-08-27 (`optimise_design.md` §11.5). Three
# contracts: the two Wilson self-field fits against their PROCESS staticmethods, and the
# whole block against `waveform` + PROCESS's own two `peak_b_field_at_pf_coil` calls.

_R_CS_INNER = 2.003843190236783
_R_CS_OUTER = 2.550659784225536
_Z_CS_UPPER = 7.936395447714745
_Z_CS_MIDDLE = 0.0

_CS_COIL = CSCoil(cs_fatigue=None)


def _reference_cs_bore_magnetic_field(j_cs, r_cs_inner, r_cs_outer, dz_cs_half):
    """`CSCoil.calculate_cs_bore_magnetic_field` -- a `@staticmethod` with no `data`."""
    return CSCoil.calculate_cs_bore_magnetic_field(
        j_cs=float(j_cs),
        r_cs_inner=float(r_cs_inner),
        r_cs_outer=float(r_cs_outer),
        dz_cs_half=float(dz_cs_half),
    )


def _reference_cs_self_peak_magnetic_field(j_cs, r_cs_inner, r_cs_outer, dz_cs_half):
    """`CSCoil.calculate_cs_self_peak_magnetic_field`.

    An instance method rather than a `@staticmethod`, but it touches no `self.data` --
    only `self.calculate_cs_bore_magnetic_field`. A module-level `CSCoil` with a `None`
    fatigue sub-model is therefore a complete adapter.
    """
    return _CS_COIL.calculate_cs_self_peak_magnetic_field(
        j_cs=float(j_cs),
        r_cs_inner=float(r_cs_inner),
        r_cs_outer=float(r_cs_outer),
        dz_cs_half=float(dz_cs_half),
    )


def _reference_cs_peak_fields(
    c_pf_cs_coil_pulse_start_ma,
    c_pf_cs_coil_flat_top_ma,
    c_pf_cs_coil_pulse_end_ma,
    r_pf_coil_middle,
    z_pf_coil_middle,
    r_cs_inner,
    r_cs_outer,
    z_cs_middle,
    z_cs_upper,
    j_cs_flat_top_end,
    j_cs_pulse_start,
    rmajor,
    plasma_current,
):
    """`ohcalc`'s field block, driven through PROCESS's own routines.

    `PFCoil.waveform`, then `peak_b_field_at_pf_coil` at `n_coil = n_cs_pf_coils`,
    `n_coil_group = 99` and time points 5 and 2 -- exactly the two calls `ohcalc` makes
    (`pfcoil.py:3344-3350`, `:3382-3388`) -- then the four combinations it forms
    (`:3352-3357`, `:3362`, `:3377-3378`, `:3390-3396`).

    **No CS filament state is seeded**, unlike `_reference_pf_coil_peak_fields` above:
    the `n_coil == n_cs_pf_coils` path takes `kk = 0` and never reads
    `.pf_coil.r/z_pf_cs_current_filaments` at all. The arrays are still allocated,
    because `xind[:kk]` indexes them, and `nfxf` is set for the same reason.
    """
    data = DataStructure()
    p = data.pf_coil
    data.build.iohcl = 1
    p.n_cs_pf_coils = N_CS_PF_COILS
    p.n_pf_coil_groups = N_PF_GROUPS
    p.n_pf_coils_in_group = np.array([*N_COILS_IN_GROUP, 1, 0, 0, 0, 0, 0, 0, 0])
    p.n_cs_current_filaments = N_CS_FILAMENTS
    p.nfxf = NFXF
    p.r_pf_cs_current_filaments = np.zeros(NGC2)
    p.z_pf_cs_current_filaments = np.zeros(NGC2)
    p.c_pf_cs_current_filaments = np.zeros(NGC2)
    p.xind = np.zeros(NGC2)

    for name, value in (
        ("c_pf_cs_coil_pulse_start_ma", c_pf_cs_coil_pulse_start_ma),
        ("c_pf_cs_coil_flat_top_ma", c_pf_cs_coil_flat_top_ma),
        ("c_pf_cs_coil_pulse_end_ma", c_pf_cs_coil_pulse_end_ma),
    ):
        full = np.zeros(NGC2)
        full[:N_CS_PF_COILS] = np.asarray(value, dtype=float)
        setattr(p, name, full)

    p.r_pf_coil_middle = np.zeros(NGC2)
    p.r_pf_coil_middle[:N_PF_COILS] = np.asarray(r_pf_coil_middle, dtype=float)
    p.z_pf_coil_middle = np.zeros(NGC2)
    p.z_pf_coil_middle[:N_PF_COILS] = np.asarray(z_pf_coil_middle, dtype=float)
    p.z_pf_coil_middle[N_PF_COILS] = float(z_cs_middle)
    p.r_pf_coil_inner = np.zeros(NGC2)
    p.r_pf_coil_inner[N_PF_COILS] = float(r_cs_inner)
    p.r_pf_coil_outer = np.zeros(NGC2)
    p.r_pf_coil_outer[N_PF_COILS] = float(r_cs_outer)

    p.j_cs_pulse_start = float(j_cs_pulse_start)
    p.j_cs_flat_top_end = float(j_cs_flat_top_end)
    p.a_cs_poloidal = _A_CS_POLOIDAL
    data.physics.rmajor = float(rmajor)
    data.physics.plasma_current = float(plasma_current)

    model = PFCoil(cs_fatigue=None, cs_coil=None)
    model.data = data
    model.waveform()

    b_self_eof = _reference_cs_self_peak_magnetic_field(
        j_cs_flat_top_end, r_cs_inner, r_cs_outer, z_cs_upper
    )
    _, _, bz_in_eof, bz_out_eof = peak_b_field_at_pf_coil(
        n_coil=N_CS_PF_COILS, n_coil_group=99, t_b_field_peak=5, data=data
    )
    b_flat_top_end = abs(bz_in_eof - b_self_eof)
    bohco = abs(bz_out_eof)

    b_self_bop = _reference_cs_self_peak_magnetic_field(
        j_cs_pulse_start, r_cs_inner, r_cs_outer, z_cs_upper
    )
    _, _, bz_in_bop, bz_out_bop = peak_b_field_at_pf_coil(
        n_coil=N_CS_PF_COILS, n_coil_group=99, t_b_field_peak=2, data=data
    )
    b_pulse_start = abs(b_self_bop + bz_in_bop)

    return (
        b_flat_top_end,
        b_pulse_start,
        0.0,
        max(b_flat_top_end, abs(b_pulse_start)),
        max(bohco, abs(bz_out_bop)),
    )


class TestCalculateCSBoreMagneticField(Tier1Contract):
    """`calculate_cs_bore_magnetic_field` -> the `@staticmethod` it ports."""

    audit_record = "models/pfcoil/fields.md"
    reference = _reference_cs_bore_magnetic_field
    ported = calculate_cs_bore_magnetic_field

    samples = [
        legacy_sample(
            "large-tokamak-converged-flat-top-end",
            j_cs=_J_CS_FLAT_TOP_END,
            r_cs_inner=_R_CS_INNER,
            r_cs_outer=_R_CS_OUTER,
            dz_cs_half=_Z_CS_UPPER,
        ),
    ]

    fuzz_bounds = {
        "j_cs": _around(_J_CS_FLAT_TOP_END, 0.30),
        "r_cs_inner": _around(_R_CS_INNER, 0.20),
        "r_cs_outer": _around(_R_CS_OUTER, 0.20),
        "dz_cs_half": _around(_Z_CS_UPPER, 0.30),
    }


class TestCalculateCSSelfPeakMagneticField(Tier1Contract):
    """`calculate_cs_self_peak_magnetic_field` -> Wilson's five-branch fit.

    **The legacy points walk all five branches.** `beta = dz_cs_half / r_cs_inner` is
    `3.96` at the converged machine, i.e. the `beta > 3` arm, and the fuzz bounds above
    cannot reach the other four (they would need the CS to be four times squatter than
    it is). Four hand-built points put `beta` at `2.5`, `1.5`, `0.9` and `0.6` by
    shortening the coil, so every arm of the reproduced `jnp.where` chain is diffed
    against the `if`/`elif` it ports -- including the two the reference machine never
    visits.
    """

    audit_record = "models/pfcoil/fields.md"
    reference = _reference_cs_self_peak_magnetic_field
    ported = calculate_cs_self_peak_magnetic_field

    samples = [
        legacy_sample(
            "large-tokamak-converged-beta-3.96",
            j_cs=_J_CS_FLAT_TOP_END,
            r_cs_inner=_R_CS_INNER,
            r_cs_outer=_R_CS_OUTER,
            dz_cs_half=_Z_CS_UPPER,
        ),
        *[
            legacy_sample(
                f"squat-cs-beta-{beta}",
                j_cs=_J_CS_FLAT_TOP_END,
                r_cs_inner=_R_CS_INNER,
                r_cs_outer=_R_CS_OUTER,
                dz_cs_half=beta * _R_CS_INNER,
            )
            for beta in (2.5, 1.5, 0.9, 0.6)
        ],
    ]

    fuzz_bounds = {
        "j_cs": _around(_J_CS_FLAT_TOP_END, 0.30),
        "r_cs_inner": _around(_R_CS_INNER, 0.20),
        "r_cs_outer": _around(_R_CS_OUTER, 0.20),
        "dz_cs_half": (0.4 * _R_CS_INNER, 5.0 * _R_CS_INNER),
    }


class TestCalculateCSPeakFields(Tier1Contract):
    """`calculate_cs_peak_fields` -> `waveform` + PROCESS's two CS-path calls."""

    audit_record = "models/pfcoil/fields.md"
    reference = _reference_cs_peak_fields
    ported = calculate_cs_peak_fields

    samples = [
        legacy_sample(
            "large-tokamak-converged",
            c_pf_cs_coil_pulse_start_ma=_C_START,
            c_pf_cs_coil_flat_top_ma=_C_FLAT,
            c_pf_cs_coil_pulse_end_ma=_C_END,
            r_pf_coil_middle=_R_MID[:N_PF_COILS],
            z_pf_coil_middle=_Z_MID[:N_PF_COILS],
            r_cs_inner=_R_CS_INNER,
            r_cs_outer=_R_CS_OUTER,
            z_cs_middle=_Z_CS_MIDDLE,
            z_cs_upper=_Z_CS_UPPER,
            j_cs_flat_top_end=_J_CS_FLAT_TOP_END,
            j_cs_pulse_start=_J_CS_PULSE_START,
            rmajor=_RMAJOR,
            plasma_current=_PLASMA_CURRENT,
        ),
    ]

    fuzz_bounds = {
        "c_pf_cs_coil_pulse_start_ma": _around(_C_START, 0.15),
        "c_pf_cs_coil_flat_top_ma": _around(_C_FLAT, 0.15),
        "c_pf_cs_coil_pulse_end_ma": _around(_C_END, 0.15),
        "r_pf_coil_middle": _around(_R_MID[:N_PF_COILS], 0.10),
        "z_pf_coil_middle": _around(_Z_MID[:N_PF_COILS], 0.10),
        "r_cs_inner": _around(_R_CS_INNER, 0.10),
        "r_cs_outer": _around(_R_CS_OUTER, 0.10),
        # Not fuzzed off zero: `z_cs_middle` is the CS's mid-height and the geometry
        # puts it at the midplane by construction (`z_cs_upper = -z_cs_lower`).
        "z_cs_middle": (-0.05, 0.05),
        "z_cs_upper": _around(_Z_CS_UPPER, 0.15),
        "j_cs_flat_top_end": _around(_J_CS_FLAT_TOP_END, 0.15),
        "j_cs_pulse_start": _around(_J_CS_PULSE_START, 0.15),
        "rmajor": _around(_RMAJOR, 0.10),
        "plasma_current": _around(_PLASMA_CURRENT, 0.15),
    }
