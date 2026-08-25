"""Harness cases for the five pure formulas ported in `pure_formulas.py`.

Registry unit #9, chunk A. All five reference adapters call the real PROCESS function
directly (four are already plain `@staticmethod`s / a bare module function -- no
`DataStructure` back door to close, so no adapter has to build one).
"""

from functional_process._harness import Tier1Contract, fuzz_samples, legacy_sample
from functional_process.models.physics.pure_formulas import (
    calaculate_stored_thermal_energy,
    calculate_total_plasma_heating_power,
    fast_alpha_beta,
    phyaux,
    rether,
)
from process.models.physics.physics import Physics, PlasmaBeta
from process.models.physics.physics import rether as _reference_rether


class TestRether(Tier1Contract):
    """`physics.rether` -> `rether`."""

    audit_record = "models/physics/pure_formulas.md"
    reference = _reference_rether
    ported = rether

    samples = [
        # tests/unit/models/physics/test_physics.py::test_rether, verbatim.
        legacy_sample(
            "test_rether",
            alphan=1.0,
            alphat=1.45,
            nd_plasma_electrons_vol_avg=7.5e19,
            dlamie=17.81065204,
            te=12.0,
            temp_plasma_ion_vol_avg_kev=13.0,
            n_charge_plasma_effective_mass_weighted_vol_avg=0.43258985,
        ),
        *fuzz_samples(
            {
                "alphan": (0.1, 2.5),
                "alphat": (0.1, 2.5),
                "nd_plasma_electrons_vol_avg": (2.0e19, 1.0e21),
                "dlamie": (14.0, 20.0),
                "te": (1.0, 40.0),
                "temp_plasma_ion_vol_avg_kev": (1.0, 40.0),
                "n_charge_plasma_effective_mass_weighted_vol_avg": (0.1, 1.0),
            },
            count=5,
            seed=0,
        ),
    ]


class TestPhyaux(Tier1Contract):
    """`Physics.phyaux` -> `phyaux`."""

    audit_record = "models/physics/pure_formulas.md"
    reference = staticmethod(Physics.phyaux)
    ported = phyaux

    # `burnup_in == 0.0` and `nd_plasma_alphas_thermal_vol_avg == 0.0` together drive
    # `burnup` itself to exactly `0.0` (both the reference and the port compute the same
    # `0/(0 + ...)`), and `molflow_plasma_fuelling_required = rndfuel / burnup` is then a
    # genuine `0/0`. PROCESS's plain-Python division raises `ZeroDivisionError`; the
    # port's `jnp` division is IEEE-754 `0.0/0.0 == nan`, which is exactly the "return
    # non-finite instead of raising" contract `_audit/test_harness.md`'s domain-guard
    # convention asks for -- see `phyaux-no-alphas` below.
    reference_domain_errors = (ZeroDivisionError,)

    samples = [
        # tests/unit/models/physics/test_physics.py::test_phyaux, both parametrizations
        # (large_tokamak_nof.IN.DAT), verbatim.
        legacy_sample(
            "phyaux-large_tokamak_nof-0",
            tauratio=1,
            burnup_in=0,
            aspect=3,
            nd_plasma_fuel_ions_vol_avg=5.8589175702454272e19,
            nd_plasma_alphas_thermal_vol_avg=7.5e18,
            fusden_total=1.9852091609123786e17,
            fusden_alpha_total=1.973996644759543e17,
            plasma_current=18398455.678867526,
            sbar=1,
            t_energy_confinement=3.401323521525641,
            vol_plasma=1888.1711539956691,
        ),
        legacy_sample(
            "phyaux-large_tokamak_nof-1",
            tauratio=1,
            burnup_in=0,
            aspect=3,
            nd_plasma_fuel_ions_vol_avg=5.8576156204039725e19,
            nd_plasma_alphas_thermal_vol_avg=7.5e18,
            fusden_total=1.9843269653375773e17,
            fusden_alpha_total=1.9731194318497056e17,
            plasma_current=18398455.678867526,
            sbar=1,
            t_energy_confinement=3.402116961408892,
            vol_plasma=1888.1711539956691,
        ),
        # burnup_in supplied directly (skips the tauratio-computed branch).
        legacy_sample(
            "phyaux-burnup_in-supplied",
            tauratio=2.0,
            burnup_in=0.15,
            aspect=3,
            nd_plasma_fuel_ions_vol_avg=5.86e19,
            nd_plasma_alphas_thermal_vol_avg=7.5e18,
            fusden_total=1.98e17,
            fusden_alpha_total=1.97e17,
            plasma_current=1.84e7,
            sbar=1,
            t_energy_confinement=3.4,
            vol_plasma=1888.0,
        ),
        # fusden_alpha_total == 0 exactly (selects the t_alpha_confinement == 0 branch).
        legacy_sample(
            "phyaux-no-alphas",
            tauratio=1.0,
            burnup_in=0.0,
            aspect=3,
            nd_plasma_fuel_ions_vol_avg=5.86e19,
            nd_plasma_alphas_thermal_vol_avg=0.0,
            fusden_total=0.0,
            fusden_alpha_total=0.0,
            plasma_current=1.84e7,
            sbar=1,
            t_energy_confinement=3.4,
            vol_plasma=1888.0,
        ),
    ]


class TestCalculateTotalPlasmaHeatingPower(Tier1Contract):
    """`Physics.calculate_total_plasma_heating_power` -> same, unchanged."""

    audit_record = "models/physics/pure_formulas.md"
    reference = staticmethod(Physics.calculate_total_plasma_heating_power)
    ported = calculate_total_plasma_heating_power

    samples = [
        # tests/unit/models/physics/test_physics.py::test_calculate_total_plasma_heating_power
        legacy_sample(
            "test_calculate_total_plasma_heating_power",
            f_p_alpha_plasma_deposited=0.25,
            p_alpha_total_mw=40.0,
            p_non_alpha_charged_mw=3.0,
            p_plasma_ohmic_mw=5.0,
            p_hcd_injected_total_mw=7.0,
        ),
        *fuzz_samples(
            {
                "f_p_alpha_plasma_deposited": (0.0, 1.0),
                "p_alpha_total_mw": (1.0, 500.0),
                "p_non_alpha_charged_mw": (0.0, 50.0),
                "p_plasma_ohmic_mw": (0.0, 20.0),
                "p_hcd_injected_total_mw": (0.0, 100.0),
            },
            count=5,
            seed=0,
        ),
    ]


class TestCalaculateStoredThermalEnergy(Tier1Contract):
    """`Physics.calaculate_stored_thermal_energy` -> same, unchanged.

    Exercised with both the electron and the ion binding's magnitude ranges (the
    function itself is species-agnostic -- see the port's docstring), rather than
    against `ElectronThermalEnergy`/`IonThermalEnergy` directly, which only differ in
    which `VarPath`s they bind.
    """

    audit_record = "models/physics/pure_formulas.md"
    reference = staticmethod(Physics.calaculate_stored_thermal_energy)
    ported = calaculate_stored_thermal_energy

    samples = [
        # tests/unit/models/physics/test_physics.py::test_calaculate_stored_thermal_energy
        legacy_sample(
            "test_calaculate_stored_thermal_energy",
            vol_plasma=100.0,
            nd_plasma_vol_avg=1e20,
            temp_plasma_density_weighted_vol_avg_kev=13.745148298980761,
        ),
        *fuzz_samples(
            {
                "vol_plasma": (100.0, 3000.0),
                "nd_plasma_vol_avg": (2.0e19, 1.0e21),
                "temp_plasma_density_weighted_vol_avg_kev": (1.0, 40.0),
            },
            count=5,
            seed=0,
        ),
    ]


def _reference_fast_alpha_beta(**kwargs):
    """Call `PlasmaBeta.fast_alpha_beta` (a static method) through the port's signature."""
    return PlasmaBeta.fast_alpha_beta(**kwargs)


class TestFastAlphaBetaIpdg89(Tier1Contract):
    """`PlasmaBeta.fast_alpha_beta` -> `fast_alpha_beta`, `i_beta_fast_alpha=0` (IPDG89).

    Split from the Ward-scaling case (`TestFastAlphaBetaWard`) below rather than fuzzed
    over both: `i_beta_fast_alpha` is a static switch (see `FastAlphaBeta`'s docstring),
    so `static_argnames` excludes it from differentiation and each value gets its own
    contract, matching how `EcrhDensityLimit`'s precedent is exercised.
    """

    audit_record = "models/physics/pure_formulas.md"
    reference = _reference_fast_alpha_beta
    ported = fast_alpha_beta
    static_argnames = ("i_beta_fast_alpha",)

    samples = [
        legacy_sample(
            "dt-plasma",
            b_plasma_poloidal_average=0.6,
            b_plasma_toroidal_on_axis=5.5,
            nd_plasma_electrons_vol_avg=7.5e19,
            nd_plasma_fuel_ions_vol_avg=5.86e19,
            nd_plasma_ions_total_vol_avg=6.6e19,
            temp_plasma_electron_density_weighted_kev=13.0,
            temp_plasma_ion_density_weighted_kev=13.0,
            pden_alpha_total_mw=0.3,
            pden_plasma_alpha_mw=0.28,
            i_beta_fast_alpha=0,
            f_plasma_fuel_deuterium=0.5,
        ),
        *fuzz_samples(
            {
                "b_plasma_poloidal_average": (0.3, 1.5),
                "b_plasma_toroidal_on_axis": (2.0, 12.0),
                "nd_plasma_electrons_vol_avg": (2.0e19, 1.0e21),
                "nd_plasma_fuel_ions_vol_avg": (1.0e19, 8.0e20),
                "nd_plasma_ions_total_vol_avg": (1.0e19, 1.0e21),
                "temp_plasma_electron_density_weighted_kev": (1.0, 30.0),
                "temp_plasma_ion_density_weighted_kev": (1.0, 30.0),
                "pden_alpha_total_mw": (0.01, 2.0),
                "pden_plasma_alpha_mw": (0.01, 2.0),
                "f_plasma_fuel_deuterium": (0.1, 0.9),
            },
            count=5,
            seed=0,
            fixed={"i_beta_fast_alpha": 0},
        ),
    ]


class TestFastAlphaBetaWard(TestFastAlphaBetaIpdg89):
    """`PlasmaBeta.fast_alpha_beta` -> `fast_alpha_beta`, `i_beta_fast_alpha=1` (Ward)."""

    samples = [
        legacy_sample(
            "dt-plasma-ward",
            b_plasma_poloidal_average=0.6,
            b_plasma_toroidal_on_axis=5.5,
            nd_plasma_electrons_vol_avg=7.5e19,
            nd_plasma_fuel_ions_vol_avg=5.86e19,
            nd_plasma_ions_total_vol_avg=6.6e19,
            temp_plasma_electron_density_weighted_kev=13.0,
            temp_plasma_ion_density_weighted_kev=13.0,
            pden_alpha_total_mw=0.3,
            pden_plasma_alpha_mw=0.28,
            i_beta_fast_alpha=1,
            f_plasma_fuel_deuterium=0.5,
        ),
        # f_plasma_fuel_deuterium > 1.0: the "negligible alpha production" branch,
        # pden_plasma_alpha_mw == 0 exactly as PROCESS's own comment describes -- this is
        # exactly the point the safe-division guard in `fast_alpha_beta` exists for. Held
        # at 1.05, not exactly 1.0: `test_gradient_agreement` perturbs every argument by
        # a relative `epsfcn`, and exactly at the branch boundary that perturbation
        # crosses `f_plasma_fuel_deuterium < 1.0` into the *reference*'s
        # `pden_alpha_total_mw / pden_plasma_alpha_mw == 0/0`, which PROCESS's plain
        # Python division raises `ZeroDivisionError` on (not a domain error this
        # contract declares -- it only fires under perturbation, not at the sample
        # itself). 1.05 keeps the whole finite-difference stencil on one side.
        legacy_sample(
            "low-deuterium-no-alphas",
            b_plasma_poloidal_average=0.6,
            b_plasma_toroidal_on_axis=5.5,
            nd_plasma_electrons_vol_avg=7.5e19,
            nd_plasma_fuel_ions_vol_avg=5.86e19,
            nd_plasma_ions_total_vol_avg=6.6e19,
            temp_plasma_electron_density_weighted_kev=13.0,
            temp_plasma_ion_density_weighted_kev=13.0,
            pden_alpha_total_mw=0.0,
            pden_plasma_alpha_mw=0.0,
            i_beta_fast_alpha=1,
            f_plasma_fuel_deuterium=1.05,
        ),
        *fuzz_samples(
            {
                "b_plasma_poloidal_average": (0.3, 1.5),
                "b_plasma_toroidal_on_axis": (2.0, 12.0),
                "nd_plasma_electrons_vol_avg": (2.0e19, 1.0e21),
                "nd_plasma_fuel_ions_vol_avg": (1.0e19, 8.0e20),
                "nd_plasma_ions_total_vol_avg": (1.0e19, 1.0e21),
                "temp_plasma_electron_density_weighted_kev": (1.0, 30.0),
                "temp_plasma_ion_density_weighted_kev": (1.0, 30.0),
                "pden_alpha_total_mw": (0.01, 2.0),
                "pden_plasma_alpha_mw": (0.01, 2.0),
                "f_plasma_fuel_deuterium": (0.1, 0.9),
            },
            count=5,
            seed=1,
            fixed={"i_beta_fast_alpha": 1},
        ),
    ]
