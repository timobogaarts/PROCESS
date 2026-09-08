"""Harness cases for the five pure formulas ported in `pure_formulas.py`.

Registry unit #9, chunk A. All five reference adapters call the real PROCESS function
directly (four are already plain `@staticmethod`s / a bare module function -- no
`DataStructure` back door to close, so no adapter has to build one).
"""

from functional_process.cottax._harness import (
    DeclaredDeviation,
    Tier1Contract,
    Tolerance,
)
from functional_process.cottax._harness.sample_store import FROM_FILE
from functional_process.cottax.physics.pure_formulas import (
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

    samples = FROM_FILE


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

    samples = FROM_FILE


class TestCalculateTotalPlasmaHeatingPower(Tier1Contract):
    """`Physics.calculate_total_plasma_heating_power` -> same, unchanged."""

    audit_record = "models/physics/pure_formulas.md"
    reference = staticmethod(Physics.calculate_total_plasma_heating_power)
    ported = calculate_total_plasma_heating_power

    samples = FROM_FILE


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

    samples = FROM_FILE


_reference_fast_alpha_beta = PlasmaBeta.fast_alpha_beta


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

    samples = FROM_FILE


class TestFastAlphaBetaWard(TestFastAlphaBetaIpdg89):
    """`PlasmaBeta.fast_alpha_beta` -> `fast_alpha_beta`, `i_beta_fast_alpha=1` (Ward).

    **The only unit in the port with a `declared_deviation`.**
    `_fast_alpha_fraction_ward` regularises PROCESS's `sqrt(temp_sum_20 - 0.65)`
    threshold (`WARD_KINK_SMOOTHING = 1e-3`), so it deliberately does not compute
    PROCESS's expression and cannot agree to round-off. That is *declared* rather than
    absorbed by widening `value_tolerance`, because the two are different claims and only
    one is true -- see `_harness/tolerance.DeclaredDeviation` for why a widened tolerance
    would be the wrong instrument.
    """

    declared_deviation = DeclaredDeviation(
        reason=(
            "the port regularises PROCESS's square-root threshold "
            "(`WARD_KINK_SMOOTHING`) because its unbounded derivative made "
            "`stellarator_helias`'s converged/stopped outcome turn on the last bit of a "
            "Jacobian cell"
        ),
        bound=Tolerance(
            rtol=1e-5,
            atol=5.0e-5,
            reason=(
                "the measured worst case with headroom. The regularisation's largest "
                "absolute effect is `0.26 r^2 sqrt(eps/2) = 4.59e-05`, at the threshold, "
                "decaying away from it; the three samples that exercise it disagree by "
                "9.78e-10, 9.43e-09 and 4.17e-08 absolute (3.0e-07 relative at the "
                "legacy sample). `atol` is what this deviation needs and `rtol` cannot "
                "supply: PROCESS returns exactly 0 below the threshold, where no "
                "relative bound is expressible"
            ),
        ),
        record="deliberate_divergences.md#1-ward_kink_smoothing",
    )

    samples = FROM_FILE
