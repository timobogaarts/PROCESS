"""Harness cases for the ported subset of `process/models/vacuum.py` (registry #16).

Audit record: `functional_process/models/vacuum.md`. Three units:

- `TestVacuumPumpingSimple` -- `Vacuum.vacuum_simple`, tier-1.
- `TestSolveDuctDiameter` -- `Vacuum._newton_method_duct_diameter`'s inner Newton loop
  (the duct-diameter root-find, isolated), tier-2, same shape as `coils.py`'s
  `TestIntersect`.
- `TestVacuumPumpingOld` -- `Vacuum.vacuum` (the full `"old"` duct-sizing model) plus
  `Vacuum.run()`'s rounding step, tier-2.

`VacuumVessel` is out of scope (unreached on the stellarator pipeline) -- see
`vacuum.md`.
"""

from types import MappingProxyType

import jax.numpy as jnp
import numpy as np

from functional_process._harness import (
    Sample,
    Tier1Contract,
    Tier2Contract,
    fuzz_samples,
    legacy_sample,
)
from functional_process.models.vacuum import (
    XMULT,
    _solve_vacuum_pumping_old,
    _solve_vacuum_pumping_old_from_fields,
    calculate_vacuum_pumping_simple,
    duct_diameter_residual,
    solve_duct_diameter,
)
from process.core import constants
from process.core.model import DataStructure
from process.models.vacuum import Vacuum


def _reference_vacuum_pumping_simple(
    molflow_plasma_fuelling_required,
    molflow_vac_pumps,
    volflow_vac_pumps_max,
    f_a_vac_pump_port_plasma_surface,
    f_volflow_vac_pumps_impedance,
    a_plasma_surface,
    n_tf_coils,
    outgasfactor,
    pres_vv_chamber_base,
    outgasindex,
    t_plant_pulse_dwell,
):
    """Call PROCESS's `Vacuum.vacuum_simple` through the port's signature."""
    data = DataStructure()
    data.physics.molflow_plasma_fuelling_required = molflow_plasma_fuelling_required
    data.vacuum.molflow_vac_pumps = molflow_vac_pumps
    data.vacuum.volflow_vac_pumps_max = volflow_vac_pumps_max
    data.vacuum.f_a_vac_pump_port_plasma_surface = f_a_vac_pump_port_plasma_surface
    data.vacuum.f_volflow_vac_pumps_impedance = f_volflow_vac_pumps_impedance
    data.physics.a_plasma_surface = a_plasma_surface
    data.tfcoil.n_tf_coils = n_tf_coils
    data.vacuum.outgasfactor = outgasfactor
    data.vacuum.pres_vv_chamber_base = pres_vv_chamber_base
    data.vacuum.outgasindex = outgasindex
    data.times.t_plant_pulse_dwell = t_plant_pulse_dwell

    v = Vacuum()
    v.data = data
    return v.vacuum_simple(output=False)


class TestVacuumPumpingSimple(Tier1Contract):
    """`Vacuum.vacuum_simple` -> `calculate_vacuum_pumping_simple`."""

    audit_record = "models/vacuum.md"
    reference = _reference_vacuum_pumping_simple
    ported = calculate_vacuum_pumping_simple

    # tests/unit/models/test_vacuum.py::TestVacuum::test_simple_model.
    samples = [
        legacy_sample(
            "simple-model",
            molflow_plasma_fuelling_required=7.5745668997694112e22,
            molflow_vac_pumps=1.2155e22,
            volflow_vac_pumps_max=27.3,
            f_a_vac_pump_port_plasma_surface=0.0203,
            f_volflow_vac_pumps_impedance=0.4,
            a_plasma_surface=1500.3146527709359,
            n_tf_coils=18,
            outgasfactor=0.0235,
            pres_vv_chamber_base=0.0005,
            outgasindex=1.0,
            t_plant_pulse_dwell=500.0,
        ),
    ]

    fuzz_bounds = {
        "molflow_plasma_fuelling_required": (1.0e21, 1.0e23),
        "molflow_vac_pumps": (1.0e21, 1.0e23),
        "volflow_vac_pumps_max": (5.0, 50.0),
        "f_a_vac_pump_port_plasma_surface": (0.005, 0.05),
        "f_volflow_vac_pumps_impedance": (0.05, 0.5),
        "a_plasma_surface": (200.0, 3000.0),
        "n_tf_coils": (10.0, 60.0),
        "outgasfactor": (0.005, 0.05),
        "pres_vv_chamber_base": (1.0e-5, 1.0e-3),
        "outgasindex": (0.5, 2.0),
        "t_plant_pulse_dwell": (10.0, 2000.0),
    }


def _reference_solve_duct_diameter(l1, l2, l3, xmult_i, ceff_i, max_iter=100, tol=0.01):
    """PROCESS's own duct-diameter Newton loop, calling its `_newton_function` directly.

    `Vacuum._newton_method_duct_diameter`'s inner loop
    (`process/models/vacuum.py:469-484`) is not exposed as a standalone function -- it's
    interleaved with the outer area-fit loop `solve_duct_geometry` ports separately. This
    adapter is a thin, faithful re-orchestration of just that inner loop (fixed `d=1.0`
    start, up to `max_iter` steps, early stop at relative step `<= tol`), but the actual
    arithmetic at every step comes from PROCESS's own `Vacuum._newton_function`
    (`@staticmethod`, called directly, not reimplemented) -- same division of labour as
    `divertor.py`'s reference adapters, which construct a `DataStructure` and call the
    real PROCESS method rather than recomputing its formula.
    """
    d = 1.0
    for _ in range(max_iter):
        d_new, _a1 = Vacuum._newton_function(d, l1, l2, l3, xmult_i, ceff_i)
        step = abs((d - d_new) / d)
        d = d_new
        if step <= tol:
            break
    return d


def _duct_diameter_residual_for_contract(solution, l1, l2, l3, xmult_i, ceff_i):
    """`Tier2Contract.residual`'s `(solution, **kwargs) -> array` shape."""
    return duct_diameter_residual(solution, l1, l2, l3, xmult_i, ceff_i)


def _duct_diameter_samples():
    """Fuzzed duct geometries, plus the one lifted from `test_old_model`'s solve.

    `tests/unit/models/test_vacuum.py::TestVacuum::test_old_model`'s helium-species
    (`i=2`) Newton solve -- extracted by instrumenting
    `Vacuum._newton_method_duct_diameter` directly, see `vacuum.md`'s worked example
    for the full derivation and why PROCESS's own reported diameter at that point does
    *not* zero this residual (its `0.01` step tolerance stops one iteration before the
    true root -- exactly the discrepancy this port's tighter default `tol` closes).
    """
    rng = np.random.default_rng(20260818)
    n = 24
    l1 = rng.uniform(0.5, 3.0, size=n)
    l2 = rng.uniform(0.5, 6.0, size=n)
    l3 = np.full(n, 2.0)
    xmult_i = rng.choice(np.asarray(XMULT), size=n)
    ceff_i = 10 ** rng.uniform(-1.0, 3.0, size=n)
    samples = [
        Sample(
            MappingProxyType({
                "l1": float(l1[i]),
                "l2": float(l2[i]),
                "l3": float(l3[i]),
                "xmult_i": float(xmult_i[i]),
                "ceff_i": float(ceff_i[i]),
            }),
            "synthetic",
            f"duct-{i}",
        )
        for i in range(n)
    ]
    samples.append(
        legacy_sample(
            "helium-duct-old-model",
            l1=0.4 + 0.63812,
            l2=0.4 + 4.0,
            l3=2.0,
            xmult_i=0.378,
            ceff_i=3.7718486393739226,
        )
    )
    return samples


class TestSolveDuctDiameter(Tier2Contract):
    """`solve_duct_diameter` -> the isolated duct-diameter Newton solve.

    No value-agreement test by construction (`Tier2Contract`) -- PROCESS's own
    `0.01`-relative-step stopping criterion is not a considered accuracy target (see
    `solve_duct_diameter`'s docstring and `vacuum.md`'s worked example), so its answer
    is not ground truth here any more than `intersect`'s 100-iteration fixed-Newton
    answer was for `coils.py`.
    """

    audit_record = "models/vacuum.md"
    reference = staticmethod(_reference_solve_duct_diameter)
    ported = solve_duct_diameter
    residual = staticmethod(_duct_diameter_residual_for_contract)

    samples = _duct_diameter_samples()


def _reference_vacuum_pumping_old(
    p_fusion_total_mw,
    rmajor,
    rminor,
    dsol,
    a_plasma_surface,
    vol_plasma,
    dr_shld_outboard,
    dr_shld_inboard,
    dr_tf_inboard,
    ritf,
    n_tf_coils,
    t_plant_pulse_dwell,
    n_divertors,
    qtorus,
    gasld,
    i_vac_pump_dwell,
    i_vacuum_pump_type,
    pres_vv_chamber_base,
    pres_div_chamber_burn,
    outgrat_fw,
    t_plant_pulse_coil_precharge,
):
    """Call PROCESS's `Vacuum.vacuum` through the port's (diagnostic) signature.

    Returns the same 7-tuple `_solve_vacuum_pumping_old` does: PROCESS's own five
    outputs (`pumpn` *not* rounded, matching that function), plus `imax`/`ceff_used` --
    the species that ended up governing the design, and the target conductance its
    reported diameter was actually solved for. Neither is a `data` field PROCESS
    writes, so they're recovered by instrumenting
    `Vacuum._newton_method_duct_diameter` (bound on this one instance, restored
    implicitly when the instance is discarded) to record its `(i, ceff[i])` on every
    call -- the last call before `vacuum()` returns is exactly the one that produced
    the final `dimax`/`imax`, same reasoning as `vacuum.md`'s worked example.

    `nplasma`/`temp_vv_chamber_gas_burn_end` are fixed, arbitrary values here (`1e20`
    K, `300` K) -- proven not to affect any of `vacuum()`'s five outputs, see
    `calculate_vacuum_pumping_old`'s docstring and `vacuum.md`.
    `temp_plasma_electron_vol_avg_kev` is set but never actually read on this path
    (only reachable through a non-convergence log message this instrumentation never
    triggers in-sample).
    """
    data = DataStructure()
    data.vacuum.i_vac_pump_dwell = i_vac_pump_dwell
    data.vacuum.i_vacuum_pump_type = i_vacuum_pump_type
    data.vacuum.pres_vv_chamber_base = pres_vv_chamber_base
    data.vacuum.pres_div_chamber_burn = pres_div_chamber_burn
    data.vacuum.outgrat_fw = outgrat_fw
    data.vacuum.temp_vv_chamber_gas_burn_end = 300.0
    data.times.t_plant_pulse_coil_precharge = t_plant_pulse_coil_precharge
    data.physics.p_fusion_total_mw = p_fusion_total_mw
    data.physics.temp_plasma_electron_vol_avg_kev = 15.0

    v = Vacuum()
    v.data = data
    captured = {}
    orig = v._newton_method_duct_diameter

    def _capture(d, i, s, xmult, l1, l2, l3, ntf, r0, aw, ritf_, thcsh, ceff):
        captured["imax"] = i
        captured["ceff_used"] = ceff[i]
        return orig(d, i, s, xmult, l1, l2, l3, ntf, r0, aw, ritf_, thcsh, ceff)

    v._newton_method_duct_diameter = _capture

    pumpn, nduct, dlscalc, mvdsh, dimax = v.vacuum(
        pfusmw=p_fusion_total_mw,
        r0=rmajor,
        aw=rminor,
        dsol=dsol,
        plasma_sarea=a_plasma_surface,
        plasma_vol=vol_plasma,
        thshldo=dr_shld_outboard,
        thshldi=dr_shld_inboard,
        thtf=dr_tf_inboard,
        ritf=ritf,
        n_tf_coils=n_tf_coils,
        t_plant_pulse_dwell=t_plant_pulse_dwell,
        nplasma=1.0e20,
        ndiv=n_divertors,
        qtorus=qtorus,
        gasld=gasld,
        output=False,
    )
    return pumpn, nduct, dlscalc, mvdsh, dimax, captured["imax"], captured["ceff_used"]


def _vacuum_pumping_old_residual(solution, **kwargs):
    """`Tier2Contract.residual`'s `(solution, **kwargs) -> array` shape.

    The defining equation of the whole design: the winning species' duct conductance,
    at the diameter the design actually reports, should equal the target conductance
    that diameter was solved for (`duct_diameter_residual`, reused from
    `solve_duct_diameter`'s own unit -- same equation, same residual). `l1`/`l2`/`l3`
    are recomputed from the same inputs `_solve_vacuum_pumping_old` itself derives them
    from (see that function's body) rather than threaded through as extra outputs.
    """
    _pumpn, _nduct, _dlscalc, _mvdsh, dimax, imax, ceff_used = solution
    l1 = kwargs["dr_shld_outboard"] + kwargs["dr_tf_inboard"]
    l2 = kwargs["dr_shld_outboard"] + 4.0
    l3 = 2.0
    xmult_i = jnp.asarray(XMULT)[jnp.asarray(imax).astype(int)]
    return duct_diameter_residual(dimax, l1, l2, l3, xmult_i, ceff_used)


def _vacuum_pumping_old_samples():
    """The `test_old_model` legacy point, plus fuzzed geometries at each switch combo.

    `i_vac_pump_dwell` (0/1/2) and `i_vacuum_pump_type` (0/1) are genuine switches
    (`_audit/naming_convention.md`) -- fuzzing them as continuous values between their
    bounds would draw meaningless non-integer switch settings, so each combination gets
    its own small fixed-switch fuzz batch instead (`fuzz_samples(..., fixed=...)`,
    same pattern `coils.py`'s `_intersect_samples` uses for a curated, non-CLI-driven
    sample set). Geometry bounds are centred on `test_old_model`'s own scale
    (`rmajor~8`, `rminor~3.3`, `n_tf_coils~18`) with enough spread to exercise more
    than one governing species (`imax`), verified empirically not to hit the
    "space limited" (`nflag = 1`) regime PROCESS's own duct-sizing model can enter --
    see `vacuum.md`'s open questions for why that regime is excluded here rather than
    exercised.
    """
    bounds = {
        "p_fusion_total_mw": (500.0, 4000.0),
        "rmajor": (6.0, 20.0),
        "rminor": (1.5, 5.0),
        "dsol": (0.05, 0.5),
        "a_plasma_surface": (400.0, 2500.0),
        "vol_plasma": (400.0, 4500.0),
        "dr_shld_outboard": (0.2, 0.8),
        "dr_shld_inboard": (0.05, 0.4),
        "dr_tf_inboard": (0.3, 1.2),
        "ritf": (3.0, 12.0),
        "n_tf_coils": (12.0, 40.0),
        "t_plant_pulse_dwell": (100.0, 1800.0),
        "n_divertors": (1.0, 2.0),
        "qtorus": (0.0, 0.0),
        "gasld": (1.0e-6, 1.0e-4),
        "pres_vv_chamber_base": (1.0e-5, 1.0e-3),
        "pres_div_chamber_burn": (0.1, 0.8),
        "outgrat_fw": (1.0e-9, 1.0e-7),
        "t_plant_pulse_coil_precharge": (10.0, 60.0),
    }
    samples = []
    for seed, dwell, pump_type in ((1, 0, 0), (2, 1, 1), (3, 2, 0)):
        samples += fuzz_samples(
            bounds,
            4,
            seed,
            fixed={"i_vac_pump_dwell": dwell, "i_vacuum_pump_type": pump_type},
        )
    samples.append(
        legacy_sample(
            "old-model-g-l-nb-ti",
            p_fusion_total_mw=2115.3899563651776,
            rmajor=8.1386000000000003,
            rminor=3.2664151549205331,
            dsol=0.22500000000000003,
            a_plasma_surface=1468.3151179059994,
            vol_plasma=2907.2299918381777,
            dr_shld_outboard=0.40000000000000002,
            dr_shld_inboard=0.12000000000000001,
            dr_tf_inboard=0.63812000000000002,
            ritf=3.6371848450794664,
            n_tf_coils=18,
            t_plant_pulse_dwell=1800.0,
            n_divertors=1,
            qtorus=0.0,
            gasld=2.7947500651998464e-05,
            i_vac_pump_dwell=0,
            i_vacuum_pump_type=1,
            pres_vv_chamber_base=0.00050000000000000001,
            pres_div_chamber_burn=0.35999999999999999,
            outgrat_fw=1.3000000000000001e-08,
            t_plant_pulse_coil_precharge=30.0,
        )
    )
    return samples


class TestVacuumPumpingOld(Tier2Contract):
    """`Vacuum.vacuum` (+ `Vacuum.run()`'s rounding) -> `_solve_vacuum_pumping_old`.

    No value-agreement test by construction. `vacuum.md`'s worked example shows
    PROCESS's own reported `dimax`, on the exact `test_old_model` legacy point, does
    not itself zero this unit's defining equation -- `duct_conductance(dimax, ...)`
    comes out ~0.16% away from the `ceff` it was meant to solve for, purely because
    PROCESS's own `0.01` relative-step stopping criterion exits one Newton step early.
    `calculate_vacuum_pumping_old` (the audited, five-output public function this
    file's node would wrap) is not tested directly -- it is a thin, trivially-correct
    slice-and-round wrapper over `_solve_vacuum_pumping_old`, tested here.
    """

    audit_record = "models/vacuum.md"
    reference = staticmethod(_reference_vacuum_pumping_old)
    ported = _solve_vacuum_pumping_old
    residual = staticmethod(_vacuum_pumping_old_residual)

    samples = _vacuum_pumping_old_samples()


def _reference_vacuum_pumping_old_from_fields(
    p_fusion_total_mw,
    rmajor,
    rminor,
    dr_fw_plasma_gap_inboard,
    dr_fw_plasma_gap_outboard,
    a_plasma_surface,
    vol_plasma,
    dr_shld_outboard,
    dr_shld_inboard,
    dr_tf_inboard,
    r_shld_inboard_inner,
    dr_shld_vv_gap_inboard,
    dr_vv_inboard,
    n_tf_coils,
    t_plant_pulse_dwell,
    n_divertors,
    qtorus,
    molflow_plasma_fuelling_required,
    m_fuel_amu,
    i_vac_pump_dwell,
    i_vacuum_pump_type,
    pres_vv_chamber_base,
    pres_div_chamber_burn,
    outgrat_fw,
    t_plant_pulse_coil_precharge,
):
    """`_reference_vacuum_pumping_old`, with `dsol`/`ritf`/`gasld` derived from the raw
    fields exactly as `Vacuum.run()` does -- what confirms `vacuum.py`'s
    `_derive_vacuum_pumping_old_locals` inlining is exact, not merely plausible.
    """
    dsol = 0.5 * (dr_fw_plasma_gap_inboard + dr_fw_plasma_gap_outboard)
    ritf = r_shld_inboard_inner - dr_shld_vv_gap_inboard - dr_vv_inboard
    gasld = 2.0 * molflow_plasma_fuelling_required * m_fuel_amu * constants.UMASS
    return _reference_vacuum_pumping_old(
        p_fusion_total_mw,
        rmajor,
        rminor,
        dsol,
        a_plasma_surface,
        vol_plasma,
        dr_shld_outboard,
        dr_shld_inboard,
        dr_tf_inboard,
        ritf,
        n_tf_coils,
        t_plant_pulse_dwell,
        n_divertors,
        qtorus,
        gasld,
        i_vac_pump_dwell,
        i_vacuum_pump_type,
        pres_vv_chamber_base,
        pres_div_chamber_burn,
        outgrat_fw,
        t_plant_pulse_coil_precharge,
    )


def _vacuum_pumping_old_from_fields_samples():
    """Field-level cousin of `_vacuum_pumping_old_samples`: same geometry/switch scale,
    `dsol`/`ritf`/`gasld` replaced by the raw fields they're derived from. The legacy
    point's raw fields are chosen so the derived locals land close to
    `_vacuum_pumping_old_samples`'s own legacy point (not required to match exactly --
    the reference is computed fresh from these fields either way).
    """
    bounds = {
        "p_fusion_total_mw": (500.0, 4000.0),
        "rmajor": (6.0, 20.0),
        "rminor": (1.5, 5.0),
        "dr_fw_plasma_gap_inboard": (0.02, 0.5),
        "dr_fw_plasma_gap_outboard": (0.02, 0.5),
        "a_plasma_surface": (400.0, 2500.0),
        "vol_plasma": (400.0, 4500.0),
        "dr_shld_outboard": (0.2, 0.8),
        "dr_shld_inboard": (0.05, 0.4),
        "dr_tf_inboard": (0.3, 1.2),
        "r_shld_inboard_inner": (3.0, 12.0),
        "dr_shld_vv_gap_inboard": (0.0, 0.0),
        "dr_vv_inboard": (0.0, 0.0),
        "n_tf_coils": (12.0, 40.0),
        "t_plant_pulse_dwell": (100.0, 1800.0),
        "n_divertors": (1.0, 2.0),
        "qtorus": (0.0, 0.0),
        "molflow_plasma_fuelling_required": (1.0e21, 5.0e22),
        "m_fuel_amu": (2.0, 3.0),
        "pres_vv_chamber_base": (1.0e-5, 1.0e-3),
        "pres_div_chamber_burn": (0.1, 0.8),
        "outgrat_fw": (1.0e-9, 1.0e-7),
        "t_plant_pulse_coil_precharge": (10.0, 60.0),
    }
    samples = []
    for seed, dwell, pump_type in ((11, 0, 0), (12, 1, 1)):
        samples += fuzz_samples(
            bounds,
            4,
            seed,
            fixed={"i_vac_pump_dwell": dwell, "i_vacuum_pump_type": pump_type},
        )
    samples.append(
        legacy_sample(
            "old-model-g-l-nb-ti-from-fields",
            p_fusion_total_mw=2115.3899563651776,
            rmajor=8.1386000000000003,
            rminor=3.2664151549205331,
            dr_fw_plasma_gap_inboard=0.22500000000000003,
            dr_fw_plasma_gap_outboard=0.22500000000000003,
            a_plasma_surface=1468.3151179059994,
            vol_plasma=2907.2299918381777,
            dr_shld_outboard=0.40000000000000002,
            dr_shld_inboard=0.12000000000000001,
            dr_tf_inboard=0.63812000000000002,
            r_shld_inboard_inner=3.8621848450794664,
            dr_shld_vv_gap_inboard=0.155,
            dr_vv_inboard=0.07,
            n_tf_coils=18,
            t_plant_pulse_dwell=1800.0,
            n_divertors=1,
            qtorus=0.0,
            molflow_plasma_fuelling_required=3.3658206e21,
            m_fuel_amu=2.5,
            i_vac_pump_dwell=0,
            i_vacuum_pump_type=1,
            pres_vv_chamber_base=0.00050000000000000001,
            pres_div_chamber_burn=0.35999999999999999,
            outgrat_fw=1.3000000000000001e-08,
            t_plant_pulse_coil_precharge=30.0,
        )
    )
    return samples


class TestVacuumPumpingOldFromFields(Tier2Contract):
    """`_solve_vacuum_pumping_old_from_fields`: the same algorithm as
    `TestVacuumPumpingOld`, exercised through the raw `.build.*`/`.physics.*` signature
    `VacuumOld`'s node actually uses, instead of PROCESS's own `dsol`/`ritf`/`gasld`-
    taking `vacuum()` boundary. Same residual, same tolerance -- this class exists to
    confirm `_derive_vacuum_pumping_old_locals`'s inlining is exact, not to re-test the
    duct-sizing algorithm itself (that's `TestVacuumPumpingOld`'s job).
    """

    audit_record = "models/vacuum.md"
    reference = staticmethod(_reference_vacuum_pumping_old_from_fields)
    ported = _solve_vacuum_pumping_old_from_fields
    residual = staticmethod(_vacuum_pumping_old_residual)

    samples = _vacuum_pumping_old_from_fields_samples()
