"""The coils' superconductor properties -- `superconpf`'s ITER Nb3Sn and NbTi arms."""

from cottax.assemblies.jax import V

from functional_process.cottax.paths import pf_coil
from functional_process.models.pfcoil.superconductor import (
    calculate_cs_critical_current_densities,
    calculate_cs_critical_current_density_iter_nb3sn,
    calculate_cs_critical_current_density_wst_nb3sn,
    calculate_cs_strand_critical_current_density,  # noqa: F401 -- re-exported for tests
    calculate_cs_temperature_margin_from_full_width_current,
    calculate_cs_temperature_margin_iter_nb3sn,
    calculate_cs_temperature_margin_wst_nb3sn,
    calculate_pf_strand_critical_current_density,
    calculate_pf_strand_critical_current_density_hazelton_zhai_rebco,  # noqa: F401 -- re-exported for tests
    calculate_pf_strand_critical_current_density_hazelton_zhai_rebco_bound,
    calculate_pf_strand_critical_current_density_hazelton_zhai_rebco_topology,  # noqa: F401 -- re-exported for tests
)


def _cs_critical_current_densities(
    critical_surface,
    b_cs_peak_flat_top_end,
    b_cs_peak_pulse_start,
    f_a_cs_void,
    fcuohsu,
    str_cs_con_res,
    temp_cs_superconductor_operating,
    a_cs_cable_space,
    a_cs_poloidal,
):
    return calculate_cs_critical_current_densities(
        critical_surface=critical_surface,
        b_cs_peak_flat_top_end=b_cs_peak_flat_top_end,
        b_cs_peak_pulse_start=b_cs_peak_pulse_start,
        f_a_cs_void=f_a_cs_void,
        fcuohsu=fcuohsu,
        str_cs_con_res=str_cs_con_res,
        temp_cs_superconductor_operating=temp_cs_superconductor_operating,
        a_cs_cable_space=a_cs_cable_space,
        a_cs_poloidal=a_cs_poloidal,
    )


def _cs_critical_current_densities_iter_nb3sn(
    b_cs_peak_flat_top_end,
    b_cs_peak_pulse_start,
    f_a_cs_void,
    fcuohsu,
    str_cs_con_res,
    temp_cs_superconductor_operating,
    a_cs_cable_space,
    a_cs_poloidal,
):
    return _cs_critical_current_densities(
        calculate_cs_critical_current_density_iter_nb3sn,
        b_cs_peak_flat_top_end,
        b_cs_peak_pulse_start,
        f_a_cs_void,
        fcuohsu,
        str_cs_con_res,
        temp_cs_superconductor_operating,
        a_cs_cable_space,
        a_cs_poloidal,
    )


def _cs_critical_current_densities_wst_nb3sn(
    b_cs_peak_flat_top_end,
    b_cs_peak_pulse_start,
    f_a_cs_void,
    fcuohsu,
    str_cs_con_res,
    temp_cs_superconductor_operating,
    a_cs_cable_space,
    a_cs_poloidal,
):
    return _cs_critical_current_densities(
        calculate_cs_critical_current_density_wst_nb3sn,
        b_cs_peak_flat_top_end,
        b_cs_peak_pulse_start,
        f_a_cs_void,
        fcuohsu,
        str_cs_con_res,
        temp_cs_superconductor_operating,
        a_cs_cable_space,
        a_cs_poloidal,
    )


_CS_CRITICAL_CURRENT_DENSITIES = (
    "j_cs_critical_flat_top_end",
    "j_cs_critical_pulse_start",
    "j_cs_conductor_critical_flat_top_end",
    "j_cs_conductor_critical_pulse_start",
    "j_crit_str_cs",
)


# cottax node: `.tokamak.cs_coil.critical_current`, `i_cs_superconductor == 1`.
CSCriticalCurrentDensitiesIterNb3Sn = pf_coil.function(
    _cs_critical_current_densities_iter_nb3sn,
    *_CS_CRITICAL_CURRENT_DENSITIES,
)


# cottax node: `.tokamak.cs_coil.critical_current`, `i_cs_superconductor == 5`.
CSCriticalCurrentDensitiesWstNb3Sn = pf_coil.function(
    _cs_critical_current_densities_wst_nb3sn,
    *_CS_CRITICAL_CURRENT_DENSITIES,
)


def _cs_temperature_margin(
    critical_surface,
    b_cs_peak_flat_top_end,
    b_cs_peak_pulse_start,
    c_pf_cs_coils_peak_ma,
    a_cs_cable_space,
    f_a_cs_void,
    fcuohsu,
    str_cs_con_res,
    temp_cs_superconductor_operating,
):
    return calculate_cs_temperature_margin_from_full_width_current(
        critical_surface=critical_surface,
        b_cs_peak_flat_top_end=b_cs_peak_flat_top_end,
        b_cs_peak_pulse_start=b_cs_peak_pulse_start,
        c_pf_cs_coils_peak_ma=c_pf_cs_coils_peak_ma,
        a_cs_cable_space=a_cs_cable_space,
        f_a_cs_void=f_a_cs_void,
        fcuohsu=fcuohsu,
        str_cs_con_res=str_cs_con_res,
        temp_cs_superconductor_operating=temp_cs_superconductor_operating,
    )


def _cs_temperature_margin_iter_nb3sn(
    b_cs_peak_flat_top_end,
    b_cs_peak_pulse_start,
    c_pf_cs_coils_peak_ma,
    a_cs_cable_space,
    f_a_cs_void,
    fcuohsu,
    str_cs_con_res,
    temp_cs_superconductor_operating,
):
    return _cs_temperature_margin(
        calculate_cs_temperature_margin_iter_nb3sn,
        b_cs_peak_flat_top_end,
        b_cs_peak_pulse_start,
        c_pf_cs_coils_peak_ma,
        a_cs_cable_space,
        f_a_cs_void,
        fcuohsu,
        str_cs_con_res,
        temp_cs_superconductor_operating,
    )


def _cs_temperature_margin_wst_nb3sn(
    b_cs_peak_flat_top_end,
    b_cs_peak_pulse_start,
    c_pf_cs_coils_peak_ma,
    a_cs_cable_space,
    f_a_cs_void,
    fcuohsu,
    str_cs_con_res,
    temp_cs_superconductor_operating,
):
    return _cs_temperature_margin(
        calculate_cs_temperature_margin_wst_nb3sn,
        b_cs_peak_flat_top_end,
        b_cs_peak_pulse_start,
        c_pf_cs_coils_peak_ma,
        a_cs_cable_space,
        f_a_cs_void,
        fcuohsu,
        str_cs_con_res,
        temp_cs_superconductor_operating,
    )


# cottax node: `.tokamak.cs_coil.temperature_margin`, `i_cs_superconductor == 1`.
CSTemperatureMarginIterNb3Sn = pf_coil.function(
    _cs_temperature_margin_iter_nb3sn,
    "temp_cs_superconductor_margin",
)


# cottax node: `.tokamak.cs_coil.temperature_margin`, `i_cs_superconductor == 5`.
CSTemperatureMarginWstNb3Sn = pf_coil.function(
    _cs_temperature_margin_wst_nb3sn,
    "temp_cs_superconductor_margin",
)


# cottax node: `.tokamak.pf_coil.strand_critical_current`.
PFStrandCriticalCurrentDensity = pf_coil.function(
    calculate_pf_strand_critical_current_density,
    "j_crit_str_pf",
    reads={
        "b_pf_coil_peak": V.pf_coil.b_pf_coil_peak[5],
        "bpf2": V.pf_coil.bpf2[5],
        "temp_pf_peak_field": V.tfcoil.tftmp,
    },
)


# cottax node: `.tokamak.pf_coil.strand_critical_current`, `i_pf_superconductor ==
# 9`.
PFStrandCriticalCurrentDensityHazeltonZhaiRebco = pf_coil.function(
    calculate_pf_strand_critical_current_density_hazelton_zhai_rebco_bound,
    "j_crit_str_pf",
)


# `.pf_coil.j_pf_wp_critical` -- left unowned, and named here so a reader who greps for
# it finds the reason rather than an omission. `ohcalc:3670-3675` writes the
# beginning-of-pulse winding-pack critical density into the CS slot and then copies it
# straight back out into `.pf_coil.j_cs_critical_pulse_start` (`:3677-3679`), so the CS
# slot is the same number under two names and the second name is the one every consumer
# reads. The six PF slots are `pfcoil()`'s (`:877`), and those `superconpf` calls **are**
# reached now -- `PFStrandCriticalCurrentDensity` above takes their third return for the
# last coil. Their first return is still not taken, and the reason has changed from
# "unported" to "no reader": `j_pf_wp_critical` is an array whose six entries nothing in
# this graph consumes (PROCESS itself only prints them, `outpf:2570-2603`), and owning it
# would mean six critical-surface evaluations for an output no edge leaves. It stays on
# `boundary.computed_by_process`'s list of fields PROCESS writes and the port does not,
# where it is invisible to `unproduced_but_computed` precisely because nothing reads it.
