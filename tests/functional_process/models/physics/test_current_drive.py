"""Harness cases for the ported heating-and-current-drive model.

`CurrentDrive.current_drive` writes its results onto `self.data` and returns nothing, so
the reference below builds a real `DataStructure`, sets every field the method reads for
the `i_hcd_primary = 10` / `i_hcd_secondary = 0` arm, calls the bound method and reads
the ten outputs back off `data` -- the same "close the data backdoor" technique
`test_confinement_time.py` uses for `calculate_confinement_time`.

The five sub-model constructor arguments are passed as `None`. That is not a shortcut: on
this arm `hcd_models` (`current_drive.py:1697-1771`) is a dict of *lambdas*, only one of
which is ever called, and model 10's is `eta_cd_norm_ecrh / (dene20 * rmajor)` -- it
touches no sub-model at all. Passing `None` is what proves it, and it is why this arm is
the one the port can reach without also porting `profiles.py` (see
`current_drive.md` § "A live PROCESS bug in two sibling arms").
"""

import pytest

from functional_process._harness import Tier1Contract, legacy_sample
from functional_process.cottax.physics.current_drive import (
    HcdElectricTotalIgnited,
    HcdElectricTotalNonIgnited,
    HcdInjectedPowerTotal,
    HcdPrimaryEfficiencyFreethyEcrhOMode,
    HcdPrimaryEfficiencyUserInputEcrh,
    HcdPrimaryInjectedPower,
    HcdPrimaryPowersElectronCyclotronNoSecondary,
    HcdSecondaryDrivenCurrent,
    HcdSecondaryHeatingNone,
    calculate_current_drive_ecrh_primary_no_secondary,
    calculate_current_drive_freethy_ecrh_primary_no_secondary,
    freethy_electron_cyclotron_efficiency,
)
from process.core.model import DataStructure
from process.models.physics.current_drive import CurrentDrive, ElectronCyclotron

# Fields the arm reads that are not arguments of the port, because the port does not
# compute what reads them: `temp_plasma_electron_vol_avg_kev` feeds only
# `eta_cd_dimensionless_hcd_primary`, and the two power terms only `big_q_plasma` --
# three of the source's writes that are outside this unit's declared closure (see the
# audit record's "What this unit does *not* port and why"). They are held fixed and
# non-zero so the reference does not divide by zero on its way past them.
_OUT_OF_CLOSURE = {
    "temp_plasma_electron_vol_avg_kev": 12.0,
    "p_fusion_total_mw": 2000.0,
    "p_plasma_ohmic_mw": 0.5,
}


def _reference_current_drive(**kwargs):
    """Call PROCESS's `CurrentDrive.current_drive` through the port's signature.

    Returns the same 10-tuple the port does, read back off `data` in the port's order.
    """
    data = DataStructure()

    data.current_drive.i_hcd_primary = 10
    data.current_drive.i_hcd_secondary = 0
    data.current_drive.i_hcd_calculations = 1

    data.current_drive.eta_cd_norm_ecrh = kwargs["eta_cd_norm_ecrh"]
    data.current_drive.eta_ecrh_injector_wall_plug = kwargs[
        "eta_ecrh_injector_wall_plug"
    ]
    data.current_drive.p_hcd_primary_extra_heat_mw = kwargs[
        "p_hcd_primary_extra_heat_mw"
    ]
    data.current_drive.p_hcd_secondary_injected_mw = kwargs[
        "p_hcd_secondary_injected_mw"
    ]

    data.physics.i_plasma_ignited = kwargs["i_plasma_ignited"]
    data.physics.nd_plasma_electrons_vol_avg = kwargs["nd_plasma_electrons_vol_avg"]
    data.physics.rmajor = kwargs["rmajor"]
    data.physics.plasma_current = kwargs["plasma_current"]
    data.physics.f_c_plasma_auxiliary = kwargs["f_c_plasma_auxiliary"]
    for name, value in _OUT_OF_CLOSURE.items():
        setattr(data.physics, name, value)

    model = CurrentDrive(None, None, None, None, None, None)
    model.data = data
    model.current_drive()

    return (
        data.current_drive.eta_cd_hcd_primary,
        data.current_drive.c_hcd_secondary_driven,
        data.current_drive.f_c_plasma_hcd_secondary,
        data.current_drive.p_hcd_primary_injected_mw,
        data.current_drive.p_hcd_ecrh_injected_total_mw,
        data.current_drive.p_hcd_ecrh_electric_mw,
        data.current_drive.eta_hcd_primary_injector_wall_plug,
        data.heat_transport.p_hcd_primary_electric_mw,
        data.current_drive.p_hcd_injected_total_mw,
        data.heat_transport.p_hcd_electric_total_mw,
    )


# `large_tokamak_eval.IN.DAT`'s own values wherever the file states one:
# `eta_ecrh_injector_wall_plug` (:122), `eta_cd_norm_ecrh` (:123),
# `p_hcd_primary_extra_heat_mw` (:125). The plasma quantities are a plausible
# large-tokamak operating point rather than a converged solve (`converged_sample` is not
# implemented -- `_harness/sampling.py`), on the same terms `test_confinement_time.py`'s
# `_BASE` states for itself.
_BASE = {
    "eta_cd_norm_ecrh": 0.30,
    "nd_plasma_electrons_vol_avg": 8.0e19,
    "rmajor": 9.0,
    "plasma_current": 18.0e6,
    "f_c_plasma_auxiliary": 0.08,
    "p_hcd_primary_extra_heat_mw": 75.0,
    "p_hcd_secondary_injected_mw": 0.0,
    "eta_ecrh_injector_wall_plug": 0.5,
    "i_plasma_ignited": 0,
}


def _point(**overrides):
    return dict(_BASE, **overrides)


class TestCurrentDriveEcrhPrimaryNoSecondary(Tier1Contract):
    """The composite for this arm -> `CurrentDrive.current_drive`, all ten outputs.

    All ten outputs, at the reference file's own switch combination. Samples cover both
    `i_plasma_ignited` arms -- the second exists to pin the *"fudge"* reset at
    `current_drive.py:2294-2299`, which is the only place in this unit where a switch
    changes an answer rather than a formula -- and one point with a non-zero secondary
    injected power, which is unphysical alongside `i_hcd_secondary = 0` but is exactly
    what PROCESS computes there: nothing forces the field to zero (it has no writer
    anywhere in `process/`), so the total at `:2268` genuinely reads it. A sample that
    left it at `0.0` everywhere would leave that read untested.
    """

    audit_record = "models/physics/current_drive.md"
    reference = _reference_current_drive
    ported = calculate_current_drive_ecrh_primary_no_secondary

    static_argnames = ("i_plasma_ignited",)

    samples = [
        legacy_sample("large-tokamak-eval-operating-point", **_point()),
        legacy_sample("ignited-wall-plug-reset", **_point(i_plasma_ignited=1)),
        legacy_sample(
            "secondary-injected-power-nonzero",
            **_point(p_hcd_secondary_injected_mw=30.0),
        ),
        legacy_sample(
            "small-machine-high-efficiency",
            **_point(
                rmajor=3.0,
                nd_plasma_electrons_vol_avg=2.0e20,
                plasma_current=6.0e6,
                eta_cd_norm_ecrh=0.5,
                eta_ecrh_injector_wall_plug=0.8,
                p_hcd_primary_extra_heat_mw=10.0,
            ),
        ),
    ]

    fuzz_fixed = {"i_plasma_ignited": 0}
    fuzz_bounds = {
        "eta_cd_norm_ecrh": (0.1, 0.6),
        "nd_plasma_electrons_vol_avg": (1.0e19, 2.0e20),
        "rmajor": (2.0, 20.0),
        "plasma_current": (1.0e6, 2.0e7),
        "f_c_plasma_auxiliary": (0.01, 0.3),
        "p_hcd_primary_extra_heat_mw": (1.0, 200.0),
        "p_hcd_secondary_injected_mw": (1.0, 50.0),
        "eta_ecrh_injector_wall_plug": (0.2, 0.8),
    }


def _reference_current_drive_freethy(**kwargs):
    """Call PROCESS's `CurrentDrive.current_drive` on the `i_hcd_primary = 13` arm.

    Same technique as `_reference_current_drive`, two differences: the switch is `13`
    and the second constructor argument is a real `ElectronCyclotron` -- model 13's
    lambda calls `self.electron_cyclotron.electron_cyclotron_freethy(...)`
    (`current_drive.py:1759-1770`), a `@staticmethod` reached *through the attribute*,
    so `None` would `AttributeError` before the static nature of the method could save
    it. `ElectronCyclotron(None)` is still the proof that no profile machinery is
    touched: its `plasma_profile` is `None` and the Freethy model never reaches it.
    """
    data = DataStructure()

    data.current_drive.i_hcd_primary = 13
    data.current_drive.i_hcd_secondary = 0
    data.current_drive.i_hcd_calculations = 1
    data.current_drive.i_ecrh_wave_mode = kwargs["i_ecrh_wave_mode"]

    data.current_drive.n_ecrh_harmonic = kwargs["n_ecrh_harmonic"]
    data.current_drive.feffcd = kwargs["feffcd"]
    data.current_drive.eta_ecrh_injector_wall_plug = kwargs[
        "eta_ecrh_injector_wall_plug"
    ]
    data.current_drive.p_hcd_primary_extra_heat_mw = kwargs[
        "p_hcd_primary_extra_heat_mw"
    ]
    data.current_drive.p_hcd_secondary_injected_mw = kwargs[
        "p_hcd_secondary_injected_mw"
    ]

    data.physics.i_plasma_ignited = kwargs["i_plasma_ignited"]
    data.physics.temp_plasma_electron_vol_avg_kev = kwargs[
        "temp_plasma_electron_vol_avg_kev"
    ]
    data.physics.n_charge_plasma_effective_vol_avg = kwargs[
        "n_charge_plasma_effective_vol_avg"
    ]
    data.physics.nd_plasma_electrons_vol_avg = kwargs["nd_plasma_electrons_vol_avg"]
    data.physics.rmajor = kwargs["rmajor"]
    data.physics.b_plasma_toroidal_on_axis = kwargs["b_plasma_toroidal_on_axis"]
    data.physics.plasma_current = kwargs["plasma_current"]
    data.physics.f_c_plasma_auxiliary = kwargs["f_c_plasma_auxiliary"]
    # `temp_plasma_electron_vol_avg_kev` is a real input on this arm (the Freethy
    # efficiency reads it), unlike arm 10's `_OUT_OF_CLOSURE` treatment; only the two
    # `big_q_plasma` feeds stay out-of-closure here.
    data.physics.p_fusion_total_mw = _OUT_OF_CLOSURE["p_fusion_total_mw"]
    data.physics.p_plasma_ohmic_mw = _OUT_OF_CLOSURE["p_plasma_ohmic_mw"]

    model = CurrentDrive(None, ElectronCyclotron(None), None, None, None, None)
    model.data = data
    model.current_drive()

    return (
        data.current_drive.eta_cd_hcd_primary,
        data.current_drive.c_hcd_secondary_driven,
        data.current_drive.f_c_plasma_hcd_secondary,
        data.current_drive.p_hcd_primary_injected_mw,
        data.current_drive.p_hcd_ecrh_injected_total_mw,
        data.current_drive.p_hcd_ecrh_electric_mw,
        data.current_drive.eta_hcd_primary_injector_wall_plug,
        data.heat_transport.p_hcd_primary_electric_mw,
        data.current_drive.p_hcd_injected_total_mw,
        data.heat_transport.p_hcd_electric_total_mw,
    )


# `spherical_tokamak_eval.IN.DAT`'s own values wherever the file states one:
# `n_ecrh_harmonic` (:129), `i_ecrh_wave_mode` (:130), `eta_ecrh_injector_wall_plug`
# (:131), `feffcd` (:132), `b_plasma_toroidal_on_axis` (:262),
# `nd_plasma_electrons_vol_avg` (:263), `rmajor` (:291),
# `temp_plasma_electron_vol_avg_kev` (:294). `p_hcd_primary_extra_heat_mw` keeps
# PROCESS's default `0.0` (`current_drive_variables.py:235` -- neither ST file sets
# it). The remaining plasma quantities (`zeff`, the current, the auxiliary fraction)
# are computed by PROCESS, not stated in the file; plausible spherical-tokamak values
# on the same terms `_BASE` states for itself.
_FREETHY_BASE = {
    "temp_plasma_electron_vol_avg_kev": 11.814206849688595,
    "n_charge_plasma_effective_vol_avg": 2.4,
    "nd_plasma_electrons_vol_avg": 9.69888313737236e19,
    "rmajor": 4.5,
    "b_plasma_toroidal_on_axis": 3.0,
    "n_ecrh_harmonic": 2.0,
    "feffcd": 1.0,
    "plasma_current": 2.0e7,
    "f_c_plasma_auxiliary": 0.1,
    "p_hcd_primary_extra_heat_mw": 0.0,
    "p_hcd_secondary_injected_mw": 0.0,
    "eta_ecrh_injector_wall_plug": 0.45,
    "i_plasma_ignited": 0,
    "i_ecrh_wave_mode": 0,
}


def _freethy_point(**overrides):
    return dict(_FREETHY_BASE, **overrides)


class TestCurrentDriveFreethyEcrhPrimaryNoSecondary(Tier1Contract):
    """The composite for the `i_hcd_primary = 13` arm, all ten outputs.

    The switch combination `spherical_tokamak_eval.IN.DAT` and `st_regression.IN.DAT`
    hold: Freethy ECCD efficiency, no secondary, O-mode. Samples cover both
    `i_plasma_ignited` arms and a non-zero secondary injected power, on the same
    grounds as `TestCurrentDriveEcrhPrimaryNoSecondary`'s; two are this arm's own:

    - `"near-cutoff-coupling"` sits on the slope of the tanh coupling factor
      (`current_drive.py:1082-1085`) rather than in its saturated tail, so the
      gradient check exercises the one place the Freethy model's derivative has
      structure the user-input model's does not.
    - `"x-mode-transcription-check"` runs the **unbound** X-mode branch
      (`i_ecrh_wave_mode = 1`). No occupant exists for it and `indat.py` refuses it
      (`UNPORTED[("i_ecrh_wave_mode", 1)]`) because no tracked input selects it -- but
      the branch is one line inside the shared pure function, and this sample pins the
      transcription (values and gradients) so a future X-mode occupant starts from
      verified ground instead of prose.

    The fuzz bounds keep `n_ecrh_harmonic * fc` above the O-mode cut-off `fp` at every
    corner (worst corner: harmonic 1.8 x field 2.5 T against density 1.4e20 leaves the
    tanh argument positive), so no fuzz point lands in the decoupled regime where the
    efficiency underflows toward zero and the downstream division by it explodes --
    PROCESS itself would produce those same explosions, but a bounded fuzz should stay
    in the physical regime the tracked files run in.
    """

    audit_record = "models/physics/current_drive.md"
    reference = _reference_current_drive_freethy
    ported = calculate_current_drive_freethy_ecrh_primary_no_secondary

    static_argnames = ("i_plasma_ignited", "i_ecrh_wave_mode")

    samples = [
        legacy_sample("spherical-tokamak-eval-operating-point", **_freethy_point()),
        legacy_sample("ignited-wall-plug-reset", **_freethy_point(i_plasma_ignited=1)),
        legacy_sample(
            "secondary-injected-power-nonzero",
            **_freethy_point(
                p_hcd_secondary_injected_mw=30.0, p_hcd_primary_extra_heat_mw=10.0
            ),
        ),
        legacy_sample(
            "near-cutoff-coupling",
            **_freethy_point(
                nd_plasma_electrons_vol_avg=1.5e20, b_plasma_toroidal_on_axis=2.2
            ),
        ),
        legacy_sample(
            "x-mode-transcription-check", **_freethy_point(i_ecrh_wave_mode=1)
        ),
    ]

    fuzz_fixed = {"i_plasma_ignited": 0, "i_ecrh_wave_mode": 0}
    fuzz_bounds = {
        "temp_plasma_electron_vol_avg_kev": (5.0, 20.0),
        "n_charge_plasma_effective_vol_avg": (1.5, 3.5),
        "nd_plasma_electrons_vol_avg": (4.0e19, 1.4e20),
        "rmajor": (3.0, 6.0),
        "b_plasma_toroidal_on_axis": (2.5, 3.8),
        "n_ecrh_harmonic": (1.8, 2.2),
        "feffcd": (0.5, 1.0),
        "plasma_current": (5.0e6, 2.5e7),
        "f_c_plasma_auxiliary": (0.01, 0.3),
        "p_hcd_primary_extra_heat_mw": (1.0, 100.0),
        "p_hcd_secondary_injected_mw": (1.0, 50.0),
        "eta_ecrh_injector_wall_plug": (0.3, 0.7),
    }


def _paths(node):
    return [i.var.path_str() for i in node.inputs], [
        o.var.path_str() for o in node.outputs
    ]


def test_the_ecrh_efficiency_occupant_reads_three_variables():
    """Model 10's reads are its own three, not the union of eleven models'.

    `hcd_models` is one dict indexed by `i_hcd_primary`
    (`process/models/physics/current_drive.py:1697-1771`), and a node carrying that
    switch as a static kwarg would have had to declare every lambda's reads at once --
    the plasma profiles, `dlamee`, `beta_total_vol_avg`, `n_ecrh_harmonic`,
    `i_ecrh_wave_mode`, `xi_ebw`, `feffcd`, `b_plasma_toroidal_on_axis`, and the
    temperatures each of them takes. The occupant declares three.
    """
    inputs, outputs = _paths(HcdPrimaryEfficiencyUserInputEcrh())

    assert set(inputs) == {
        ".current_drive.eta_cd_norm_ecrh",
        ".physics.nd_plasma_electrons_vol_avg",
        ".physics.rmajor",
    }
    assert outputs == [".current_drive.eta_cd_hcd_primary"]

    # `feffcd` scales every *other* model's efficiency (`:1704`, `:1713`, ... `:1769`)
    # and is the one read model 10 conspicuously does not make: `:1744-1747` has no
    # `feffcd` factor. That asymmetry is invisible in a union-of-arms node.
    assert ".current_drive.feffcd" not in inputs


def test_the_freethy_efficiency_occupant_reads_seven_variables():
    """Model 13's reads are its own seven, disjoint from model 10's knob.

    The mirror image of the model-10 test above: the Freethy occupant *does* read
    `feffcd` (its lambda multiplies by it, `current_drive.py:1759-1770`) and does *not*
    read `eta_cd_norm_ecrh`, the user-supplied gamma that is the whole of model 10.
    Two occupants of one family with a 2-of-8 overlap (`nd_plasma_electrons_vol_avg`,
    `rmajor`) is the reads-differ evidence for splitting `i_hcd_primary` per value.
    """
    inputs, outputs = _paths(HcdPrimaryEfficiencyFreethyEcrhOMode())

    assert set(inputs) == {
        ".physics.temp_plasma_electron_vol_avg_kev",
        ".physics.n_charge_plasma_effective_vol_avg",
        ".physics.rmajor",
        ".physics.nd_plasma_electrons_vol_avg",
        ".physics.b_plasma_toroidal_on_axis",
        ".current_drive.n_ecrh_harmonic",
        ".current_drive.feffcd",
    }
    assert outputs == [".current_drive.eta_cd_hcd_primary"]

    assert ".current_drive.eta_cd_norm_ecrh" not in inputs
    # The nested switch is not a port: the occupant pins O-mode, it does not read the
    # field (`_audit/naming_convention.md` § "switches are not ports").
    assert ".current_drive.i_ecrh_wave_mode" not in inputs


def test_the_wave_mode_switch_is_static_with_identical_reads():
    """`i_ecrh_wave_mode` is the static-kwarg exception, and here is the evidence.

    `traceability_policy.md` demands the reads-set comparison, not an assertion of
    equivalence: both wave modes compute from the same seven arguments -- the two
    cut-offs are formed from `fc` and `fp`, which O-mode and X-mode alike derive from
    `b_plasma_toroidal_on_axis` and `nd_plasma_electrons_vol_avg`
    (`current_drive.py:1046-1079`) -- so the same call succeeds for both modes and no
    argument can be dropped for either. An invalid mode raises the reference's own
    `ValueError`, transcribed rather than converted to a refusal, because the pure
    function ports the staticmethod defects included.
    """
    kwargs = {
        name: _FREETHY_BASE[name]
        for name in (
            "temp_plasma_electron_vol_avg_kev",
            "n_charge_plasma_effective_vol_avg",
            "rmajor",
            "nd_plasma_electrons_vol_avg",
            "b_plasma_toroidal_on_axis",
            "n_ecrh_harmonic",
            "feffcd",
        )
    }

    o_mode = float(freethy_electron_cyclotron_efficiency(**kwargs, i_ecrh_wave_mode=0))
    x_mode = float(freethy_electron_cyclotron_efficiency(**kwargs, i_ecrh_wave_mode=1))
    # Same reads, different cut-off: the values must differ (the X-mode right-hand
    # cut-off sits above the O-mode plasma-frequency cut-off, so its coupling factor is
    # smaller at this operating point) or the branch is not actually selected.
    assert o_mode != pytest.approx(x_mode, rel=0.0)

    with pytest.raises(ValueError, match="Invalid wave mode"):
        freethy_electron_cyclotron_efficiency(**kwargs, i_ecrh_wave_mode=2)

    # The occupant pins O-mode: its answer is the mode-0 branch of the pure function.
    assert float(HcdPrimaryEfficiencyFreethyEcrhOMode()(**kwargs)) == pytest.approx(
        o_mode, rel=0.0
    )


def test_the_ignition_switch_removes_two_reads_from_the_electric_total():
    """`HcdElectricTotalIgnited` reads nothing, which is the whole point of the split.

    `current_drive.py:2289-2299` computes the sum of the two systems' wall-plug powers
    and then, on an ignited plasma, throws it away for a literal `0.0`. One node
    branching internally would declare both reads on both arms; two occupants declare
    them where they are made.
    """
    non_ignited_inputs, non_ignited_outputs = _paths(HcdElectricTotalNonIgnited())
    ignited_inputs, ignited_outputs = _paths(HcdElectricTotalIgnited())

    assert set(non_ignited_inputs) == {
        ".heat_transport.p_hcd_primary_electric_mw",
        ".heat_transport.p_hcd_secondary_electric_mw",
    }
    assert ignited_inputs == []
    assert (
        non_ignited_outputs
        == ignited_outputs
        == [".heat_transport.p_hcd_electric_total_mw"]
    )

    assert HcdElectricTotalIgnited()() == pytest.approx(0.0, abs=0.0)
    assert HcdElectricTotalNonIgnited()(
        p_hcd_primary_electric_mw=219.12, p_hcd_secondary_electric_mw=0.0
    ) == pytest.approx(219.12, rel=0.0)


def test_the_no_secondary_occupant_declares_the_three_zeros(
    reads_only_its_own_statement,
):
    """The `i_hcd_secondary == 0` arm produces three zeros and reads nothing.

    PROCESS assigns exactly one of them (`p_hcd_secondary_extra_heat_mw = 0.0`,
    `current_drive.py:1682`) and leaves the other two at their `DataStructure` defaults
    because every block that would write them is guarded on a `secondary_cdm.method`
    that `NO_CURRENT_DRIVE` does not have. Declaring all three is what keeps two
    computed quantities off the boundary -- see the class docstring for why that is the
    right call rather than a convenience.
    """
    node = reads_only_its_own_statement(HcdSecondaryHeatingNone())
    inputs, outputs = _paths(node)

    assert inputs == [
        "^stated.current_drive.eta_cd_hcd_secondary",
        "^stated.current_drive.p_hcd_secondary_extra_heat_mw",
        "^stated.heat_transport.p_hcd_secondary_electric_mw",
    ]
    assert set(outputs) == {
        ".current_drive.eta_cd_hcd_secondary",
        ".current_drive.p_hcd_secondary_extra_heat_mw",
        ".heat_transport.p_hcd_secondary_electric_mw",
    }
    assert HcdSecondaryHeatingNone()(0.0, 0.0, 0.0) == (0.0, 0.0, 0.0)

    # The defaults the node is asserting. If PROCESS ever changes one, this fails here
    # rather than silently shifting a run's answer.
    data = DataStructure()
    assert data.current_drive.eta_cd_hcd_secondary == pytest.approx(0.0, abs=0.0)
    assert data.heat_transport.p_hcd_secondary_electric_mw == pytest.approx(0.0, abs=0.0)


def test_the_ecrh_primary_block_does_not_read_the_other_injectors():
    """Four injector efficiencies and every neutral-beam parameter stay undeclared.

    `current_drive.py:2068-2260` is five wall-plug blocks, one per technology, each
    reading its own `eta_*_injector_wall_plug` and the neutral-beam one reading four more
    fields besides. The occupant for the ECRH block reads one of the five.
    """
    inputs, outputs = _paths(HcdPrimaryPowersElectronCyclotronNoSecondary())

    assert set(inputs) == {
        ".current_drive.p_hcd_primary_injected_mw",
        ".current_drive.p_hcd_primary_extra_heat_mw",
        ".current_drive.eta_ecrh_injector_wall_plug",
    }
    for other in (
        ".current_drive.eta_lowhyb_injector_wall_plug",
        ".current_drive.eta_icrh_injector_wall_plug",
        ".current_drive.eta_ebw_injector_wall_plug",
        ".current_drive.eta_beam_injector_wall_plug",
        ".current_drive.e_beam_kev",
        ".current_drive.f_p_beam_orbit_loss",
        ".current_drive.f_p_beam_shine_through",
        ".current_drive.f_p_beam_injected_ions",
    ):
        assert other not in inputs

    assert set(outputs) == {
        ".current_drive.p_hcd_ecrh_injected_total_mw",
        ".current_drive.p_hcd_ecrh_electric_mw",
        ".current_drive.eta_hcd_primary_injector_wall_plug",
        ".heat_transport.p_hcd_primary_electric_mw",
    }


def test_the_three_boundary_reads_are_produced():
    """The three variables `_audit/tokamak_boundary.md` attributes to this slot.

    That file's `.tokamak.current_drive` section lists exactly these, with their readers
    in `costs`, `physics`, `power` and `availability`. This test is the statement that
    the slot's occupants now own all three -- it is what the unit was dispatched to do,
    so it is worth asserting rather than inferring from the class list.
    """
    owned = set()
    for node in (
        HcdPrimaryEfficiencyUserInputEcrh(),
        HcdSecondaryHeatingNone(),
        HcdPrimaryPowersElectronCyclotronNoSecondary(),
        HcdInjectedPowerTotal(),
        HcdElectricTotalNonIgnited(),
    ):
        owned.update(o.var.path_str() for o in node.outputs)

    assert {
        ".current_drive.p_hcd_ecrh_injected_total_mw",
        ".current_drive.p_hcd_injected_total_mw",
        ".heat_transport.p_hcd_electric_total_mw",
    } <= owned


def test_no_node_reads_what_it_owns():
    """cottax's hard error, checked here because the accumulators invite it.

    `p_hcd_ecrh_injected_total_mw` is written with `+=` in PROCESS (`:2147`), which is
    the shape that becomes a self-loop if ported literally. It is not one here: the prior
    value is the secondary system's contribution, which on this arm is the literal `0.0`
    and not a read at all.
    """
    for node in (
        HcdPrimaryEfficiencyUserInputEcrh(),
        HcdPrimaryEfficiencyFreethyEcrhOMode(),
        HcdSecondaryHeatingNone(),
        HcdSecondaryDrivenCurrent(),
        HcdPrimaryInjectedPower(),
        HcdPrimaryPowersElectronCyclotronNoSecondary(),
        HcdInjectedPowerTotal(),
        HcdElectricTotalNonIgnited(),
        HcdElectricTotalIgnited(),
    ):
        inputs, outputs = _paths(node)
        assert not set(inputs) & set(outputs), type(node).__name__


@pytest.mark.parametrize("i_plasma_ignited", [0, 1])
def test_the_nodes_compose_to_the_composite(i_plasma_ignited):
    """Calling the occupants in dependency order reproduces the composite exactly.

    The composite is what `TestCurrentDriveEcrhPrimaryNoSecondary` diffs against
    PROCESS; this is what says the node split did not change it. Without this, the
    occupants would be validated only by their declared ports, and a body that drifted
    from the pure function it wraps would pass every other test in this file.
    """
    kwargs = _point(i_plasma_ignited=i_plasma_ignited, p_hcd_secondary_injected_mw=30.0)

    eta_cd_hcd_primary = HcdPrimaryEfficiencyUserInputEcrh()(
        eta_cd_norm_ecrh=kwargs["eta_cd_norm_ecrh"],
        nd_plasma_electrons_vol_avg=kwargs["nd_plasma_electrons_vol_avg"],
        rmajor=kwargs["rmajor"],
    )
    (
        eta_cd_hcd_secondary,
        p_hcd_secondary_extra_heat_mw,
        p_hcd_secondary_electric_mw,
    ) = HcdSecondaryHeatingNone()(0.0, 0.0, 0.0)
    _, f_c_plasma_hcd_secondary = HcdSecondaryDrivenCurrent()(
        eta_cd_hcd_secondary=eta_cd_hcd_secondary,
        p_hcd_secondary_injected_mw=kwargs["p_hcd_secondary_injected_mw"],
        plasma_current=kwargs["plasma_current"],
    )
    p_hcd_primary_injected_mw = HcdPrimaryInjectedPower()(
        f_c_plasma_auxiliary=kwargs["f_c_plasma_auxiliary"],
        f_c_plasma_hcd_secondary=f_c_plasma_hcd_secondary,
        plasma_current=kwargs["plasma_current"],
        eta_cd_hcd_primary=eta_cd_hcd_primary,
    )
    (
        p_hcd_ecrh_injected_total_mw,
        _,
        _,
        p_hcd_primary_electric_mw,
    ) = HcdPrimaryPowersElectronCyclotronNoSecondary()(
        p_hcd_primary_injected_mw=p_hcd_primary_injected_mw,
        p_hcd_primary_extra_heat_mw=kwargs["p_hcd_primary_extra_heat_mw"],
        eta_ecrh_injector_wall_plug=kwargs["eta_ecrh_injector_wall_plug"],
    )
    p_hcd_injected_total_mw = HcdInjectedPowerTotal()(
        p_hcd_primary_injected_mw=p_hcd_primary_injected_mw,
        p_hcd_primary_extra_heat_mw=kwargs["p_hcd_primary_extra_heat_mw"],
        p_hcd_secondary_injected_mw=kwargs["p_hcd_secondary_injected_mw"],
        p_hcd_secondary_extra_heat_mw=p_hcd_secondary_extra_heat_mw,
    )
    if i_plasma_ignited == 0:
        p_hcd_electric_total_mw = HcdElectricTotalNonIgnited()(
            p_hcd_primary_electric_mw=p_hcd_primary_electric_mw,
            p_hcd_secondary_electric_mw=p_hcd_secondary_electric_mw,
        )
    else:
        p_hcd_electric_total_mw = HcdElectricTotalIgnited()()

    composite = calculate_current_drive_ecrh_primary_no_secondary(**kwargs)

    assert composite[0] == eta_cd_hcd_primary
    assert composite[3] == p_hcd_primary_injected_mw
    assert composite[4] == p_hcd_ecrh_injected_total_mw
    assert composite[7] == p_hcd_primary_electric_mw
    assert composite[8] == p_hcd_injected_total_mw
    assert composite[9] == p_hcd_electric_total_mw


def test_the_freethy_nodes_compose_to_the_composite():
    """The `i_hcd_primary = 13` occupant chain reproduces its composite exactly.

    Same statement as `test_the_nodes_compose_to_the_composite`, for the arm the
    spherical tokamak files select. Only the first stage differs from that test's
    chain, so only the first stage and the values threaded through the shared stages
    are re-derived here; a Freethy occupant whose body drifted from
    `freethy_electron_cyclotron_efficiency` would pass every port-declaration test in
    this file and fail this one.
    """
    kwargs = _freethy_point(p_hcd_secondary_injected_mw=30.0)

    eta_cd_hcd_primary = HcdPrimaryEfficiencyFreethyEcrhOMode()(
        temp_plasma_electron_vol_avg_kev=kwargs["temp_plasma_electron_vol_avg_kev"],
        n_charge_plasma_effective_vol_avg=kwargs["n_charge_plasma_effective_vol_avg"],
        rmajor=kwargs["rmajor"],
        nd_plasma_electrons_vol_avg=kwargs["nd_plasma_electrons_vol_avg"],
        b_plasma_toroidal_on_axis=kwargs["b_plasma_toroidal_on_axis"],
        n_ecrh_harmonic=kwargs["n_ecrh_harmonic"],
        feffcd=kwargs["feffcd"],
    )
    (
        eta_cd_hcd_secondary,
        p_hcd_secondary_extra_heat_mw,
        p_hcd_secondary_electric_mw,
    ) = HcdSecondaryHeatingNone()(0.0, 0.0, 0.0)
    _, f_c_plasma_hcd_secondary = HcdSecondaryDrivenCurrent()(
        eta_cd_hcd_secondary=eta_cd_hcd_secondary,
        p_hcd_secondary_injected_mw=kwargs["p_hcd_secondary_injected_mw"],
        plasma_current=kwargs["plasma_current"],
    )
    p_hcd_primary_injected_mw = HcdPrimaryInjectedPower()(
        f_c_plasma_auxiliary=kwargs["f_c_plasma_auxiliary"],
        f_c_plasma_hcd_secondary=f_c_plasma_hcd_secondary,
        plasma_current=kwargs["plasma_current"],
        eta_cd_hcd_primary=eta_cd_hcd_primary,
    )
    (
        p_hcd_ecrh_injected_total_mw,
        _,
        _,
        p_hcd_primary_electric_mw,
    ) = HcdPrimaryPowersElectronCyclotronNoSecondary()(
        p_hcd_primary_injected_mw=p_hcd_primary_injected_mw,
        p_hcd_primary_extra_heat_mw=kwargs["p_hcd_primary_extra_heat_mw"],
        eta_ecrh_injector_wall_plug=kwargs["eta_ecrh_injector_wall_plug"],
    )
    p_hcd_injected_total_mw = HcdInjectedPowerTotal()(
        p_hcd_primary_injected_mw=p_hcd_primary_injected_mw,
        p_hcd_primary_extra_heat_mw=kwargs["p_hcd_primary_extra_heat_mw"],
        p_hcd_secondary_injected_mw=kwargs["p_hcd_secondary_injected_mw"],
        p_hcd_secondary_extra_heat_mw=p_hcd_secondary_extra_heat_mw,
    )
    p_hcd_electric_total_mw = HcdElectricTotalNonIgnited()(
        p_hcd_primary_electric_mw=p_hcd_primary_electric_mw,
        p_hcd_secondary_electric_mw=p_hcd_secondary_electric_mw,
    )

    composite = calculate_current_drive_freethy_ecrh_primary_no_secondary(**kwargs)

    assert composite[0] == eta_cd_hcd_primary
    assert composite[3] == p_hcd_primary_injected_mw
    assert composite[4] == p_hcd_ecrh_injected_total_mw
    assert composite[7] == p_hcd_primary_electric_mw
    assert composite[8] == p_hcd_injected_total_mw
    assert composite[9] == p_hcd_electric_total_mw
