"""Harness cases for registry unit #23's two ported functions.

Both reference adapters build a **real** `DataStructure`, either loaded with PROCESS's
own shipped L(Z, Te)/Zav(Te) tables (`initialise_imprad`, run once at import) or with a
single-species slice of one, and call the **real** PROCESS functions. Nothing is
transcribed.
"""

import numpy as np

from functional_process._harness import Tier1Contract, legacy_sample
from functional_process.models.physics.impurity_radiation import (
    calculate_average_charge_at_temp,
    element2index,
)
from process.core.model import DataStructure
from process.models.physics import impurity_radiation as reference_module


def _real_tables():
    """The real 14-species tables, loaded from PROCESS's shipped data files once.

    Same reasoning as `test_radiation_power.py::_impurity_tables`: these arrays are
    compile-time constants of the graph (see the port module's docstring), so loading
    them once at import and reusing them across samples is not a shortcut.
    """
    data = DataStructure()
    reference_module.initialise_imprad(data)
    return data


_DATA = _real_tables()
_TEMP_TABLE = np.array(_DATA.impurity_radiation.temp_impurity_keV_array, dtype=float)
_ZAV_TABLE = np.array(_DATA.impurity_radiation.impurity_arr_zav, dtype=float)
_LABELS = np.array(_DATA.impurity_radiation.impurity_arr_label)

_HYDROGEN, _ARGON = 0, 8


def _reference_average_charge_at_temp(
    temp_electron_kev, temp_impurity_kev, impurity_arr_zav
):
    """`impurity_radiation.calculate_average_charge_at_temp`, one species.

    Reconstructs a single-row `DataStructure` around the given table (mirroring
    `test_radiation_power.py`'s `_reference_impurity_radiation_power_density`) so
    PROCESS's own `data`-based entry point is what actually runs, not a transcription
    of it.
    """
    temp_impurity_kev = np.atleast_1d(np.asarray(temp_impurity_kev, dtype=float))
    impurity_arr_zav = np.atleast_1d(np.asarray(impurity_arr_zav, dtype=float))
    data = DataStructure()
    data.impurity_radiation.temp_impurity_keV_array = temp_impurity_kev[np.newaxis, :]
    data.impurity_radiation.impurity_arr_zav = impurity_arr_zav[np.newaxis, :]
    data.impurity_radiation.impurity_arr_len_tab = np.array([temp_impurity_kev.size])
    return reference_module.calculate_average_charge_at_temp(
        imp_element_index=0,
        temp_electron_kev=np.atleast_1d(np.asarray(temp_electron_kev, dtype=float)),
        data=data,
    )


def _reference_element2index(element, impurity_arr_label):
    """`impurity_radiation.element2index`, label array passed in rather than fixed."""
    data = DataStructure()
    data.impurity_radiation.impurity_arr_label = np.asarray(impurity_arr_label)
    return reference_module.element2index(element, data)


class TestCalculateAverageChargeAtTemp(Tier1Contract):
    """`impurity_radiation.calculate_average_charge_at_temp`, one species.

    The two table arguments are `static_argnames`: tabulated atomic-physics constants
    loaded from `process/data/lz_non_corona_14_elements/*.dat` at startup, never written
    by any model and never an iteration variable, same reasoning
    `TestImpurityRadiationPowerDensity` in `test_radiation_power.py` already gives for
    its own table arguments. Only `temp_electron_kev`'s gradient is checked.
    """

    audit_record = "models/physics/impurity_radiation.md"
    reference = _reference_average_charge_at_temp
    ported = calculate_average_charge_at_temp
    static_argnames = ("temp_impurity_kev", "impurity_arr_zav")

    samples = [
        legacy_sample(
            # `tests/unit/models/physics/test_impurity_radiation.py::test_zav_of_te`'s
            # point verbatim, on hydrogen (index 0). Hydrogen is fully ionised at every
            # tabulated temperature, so <Z> == 1 throughout -- a real oracle, but a weak
            # exercise of the interpolation itself; `argon-varying` below covers that.
            "test_zav_of_te-hydrogen",
            temp_electron_kev=np.array([
                27.73451868,
                27.25167194,
                25.82164396,
                23.50149071,
                20.39190536,
                16.64794796,
                12.50116941,
                8.31182764,
                4.74643357,
                0.1,
            ]),
            temp_impurity_kev=_TEMP_TABLE[_HYDROGEN],
            impurity_arr_zav=_ZAV_TABLE[_HYDROGEN],
        ),
        legacy_sample(
            # Argon: partially ionised across most of the table, so <Z> varies by more
            # than an order of magnitude over this range -- the interpolation is doing
            # real work, and this is also the sample that would show a mismatch if the
            # dropped boundary clamps were not actually redundant (0.0005 keV and 45 keV
            # sit outside the table on both sides).
            "argon-varying",
            temp_electron_kev=np.array([0.0005, 0.05, 0.5, 2.0, 8.0, 20.0, 45.0]),
            temp_impurity_kev=_TEMP_TABLE[_ARGON],
            impurity_arr_zav=_ZAV_TABLE[_ARGON],
        ),
    ]

    fuzz_bounds = {
        "temp_electron_kev": (
            np.full(7, 1.0e-4),
            np.full(7, 60.0),
        ),
    }
    fuzz_fixed = {
        "temp_impurity_kev": _TEMP_TABLE[_ARGON],
        "impurity_arr_zav": _ZAV_TABLE[_ARGON],
    }


class TestElement2Index(Tier1Contract):
    """`impurity_radiation.element2index` -- a label lookup, not a numerical function.

    Both arguments are `static_argnames`: `element` is a string and `impurity_arr_label`
    is the fixed 14-entry species order, neither a quantity any solver perturbs. No
    `fuzz_bounds` -- there is no continuous domain to draw from; the 14 legacy samples
    below (PROCESS's actual species order) are the entire input space that matters.
    """

    audit_record = "models/physics/impurity_radiation.md"
    reference = _reference_element2index
    ported = element2index
    static_argnames = ("element", "impurity_arr_label")

    samples = [
        legacy_sample(
            f"species-{label.strip()}", element=label, impurity_arr_label=_LABELS
        )
        for label in _LABELS
    ]
