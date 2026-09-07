"""The coils' superconductor properties -- `superconpf`'s ITER Nb3Sn and NbTi arms."""

import equinox as eqx
from cottax.interfaces.pytree_namespace_module import (
    ExplicitFunction,
    From,
    FromExactly,
    OutputInto,
)

from functional_process.cottax.pfcoil import (
    N_PF_COILS,
    SPHERICAL_TOKAMAK_TOPOLOGY,
    PFCoilTopology,
)
from functional_process.cottax.paths import pf_coil, superconducting_tfcoil, tfcoil
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
    calculate_pf_strand_critical_current_density_hazelton_zhai_rebco_topology,
)


class CSCriticalCurrentDensitiesIterNb3Sn(ExplicitFunction):
    """cottax node: `.tokamak.cs_coil.critical_current`, `i_cs_superconductor == 1`."""

    j_cs_critical_flat_top_end = OutputInto(pf_coil)
    j_cs_critical_pulse_start = OutputInto(pf_coil)
    j_cs_conductor_critical_flat_top_end = OutputInto(pf_coil)
    j_cs_conductor_critical_pulse_start = OutputInto(pf_coil)
    j_crit_str_cs = OutputInto(pf_coil)

    _critical_surface = staticmethod(calculate_cs_critical_current_density_iter_nb3sn)
    """The arm's critical-current function."""

    def __call__(
        self,
        b_cs_peak_flat_top_end=From(pf_coil),
        b_cs_peak_pulse_start=From(pf_coil),
        f_a_cs_void=From(pf_coil),
        fcuohsu=From(pf_coil),
        str_cs_con_res=From(tfcoil),
        temp_cs_superconductor_operating=From(pf_coil),
        a_cs_cable_space=From(pf_coil),
        a_cs_poloidal=From(pf_coil),
    ):
        return calculate_cs_critical_current_densities(
            critical_surface=self._critical_surface,
            b_cs_peak_flat_top_end=b_cs_peak_flat_top_end,
            b_cs_peak_pulse_start=b_cs_peak_pulse_start,
            f_a_cs_void=f_a_cs_void,
            fcuohsu=fcuohsu,
            str_cs_con_res=str_cs_con_res,
            temp_cs_superconductor_operating=temp_cs_superconductor_operating,
            a_cs_cable_space=a_cs_cable_space,
            a_cs_poloidal=a_cs_poloidal,
        )


class CSCriticalCurrentDensitiesWstNb3Sn(CSCriticalCurrentDensitiesIterNb3Sn):
    """cottax node: `.tokamak.cs_coil.critical_current`, `i_cs_superconductor == 5`."""

    _critical_surface = staticmethod(calculate_cs_critical_current_density_wst_nb3sn)


class CSTemperatureMarginIterNb3Sn(ExplicitFunction):
    """cottax node: `.tokamak.cs_coil.temperature_margin`, `i_cs_superconductor == 1`.
    """

    temp_cs_superconductor_margin = OutputInto(pf_coil)

    _critical_surface = staticmethod(calculate_cs_temperature_margin_iter_nb3sn)

    def __call__(
        self,
        b_cs_peak_flat_top_end=From(pf_coil),
        b_cs_peak_pulse_start=From(pf_coil),
        c_pf_cs_coils_peak_ma=From(pf_coil),
        a_cs_cable_space=From(pf_coil),
        f_a_cs_void=From(pf_coil),
        fcuohsu=From(pf_coil),
        str_cs_con_res=From(tfcoil),
        temp_cs_superconductor_operating=From(pf_coil),
    ):
        return calculate_cs_temperature_margin_from_full_width_current(
            critical_surface=self._critical_surface,
            b_cs_peak_flat_top_end=b_cs_peak_flat_top_end,
            b_cs_peak_pulse_start=b_cs_peak_pulse_start,
            c_pf_cs_coils_peak_ma=c_pf_cs_coils_peak_ma,
            a_cs_cable_space=a_cs_cable_space,
            f_a_cs_void=f_a_cs_void,
            fcuohsu=fcuohsu,
            str_cs_con_res=str_cs_con_res,
            temp_cs_superconductor_operating=temp_cs_superconductor_operating,
        )


class CSTemperatureMarginWstNb3Sn(CSTemperatureMarginIterNb3Sn):
    """cottax node: `.tokamak.cs_coil.temperature_margin`, `i_cs_superconductor == 5`.
    """

    _critical_surface = staticmethod(calculate_cs_temperature_margin_wst_nb3sn)


class PFStrandCriticalCurrentDensity(ExplicitFunction):
    """cottax node: `.tokamak.pf_coil.strand_critical_current`."""

    j_crit_str_pf = OutputInto(pf_coil)

    def __call__(
        self,
        b_pf_coil_peak_last=FromExactly(pf_coil.b_pf_coil_peak[N_PF_COILS - 1]),
        bpf2_last=FromExactly(pf_coil.bpf2[N_PF_COILS - 1]),
        tftmp=From(tfcoil),
        fcupfsu=From(pf_coil),
    ):
        return calculate_pf_strand_critical_current_density(
            b_pf_coil_peak=b_pf_coil_peak_last,
            bpf2=bpf2_last,
            temp_pf_peak_field=tftmp,
            fcupfsu=fcupfsu,
        )


class PFStrandCriticalCurrentDensityHazeltonZhaiRebco(ExplicitFunction):
    """cottax node: `.tokamak.pf_coil.strand_critical_current`, `i_pf_superconductor ==
    9`.
    """

    topology: PFCoilTopology = eqx.field(static=True, default=SPHERICAL_TOKAMAK_TOPOLOGY)
    """Static -- which coil the `pfcoil()` loop finishes on, and therefore which slot of
    the two peak-field arrays the surviving scalar came from.
    """

    j_crit_str_pf = OutputInto(pf_coil)

    def __call__(
        self,
        b_pf_coil_peak=From(pf_coil),
        bpf2=From(pf_coil),
        tftmp=From(tfcoil),
        fcupfsu=From(pf_coil),
        dr_tf_hts_tape=From(superconducting_tfcoil),
        dx_tf_hts_tape_rebco=From(superconducting_tfcoil),
        dx_tf_hts_tape_total=From(superconducting_tfcoil),
    ):
        return calculate_pf_strand_critical_current_density_hazelton_zhai_rebco_topology(
            b_pf_coil_peak=b_pf_coil_peak,
            bpf2=bpf2,
            tftmp=tftmp,
            fcupfsu=fcupfsu,
            dr_tf_hts_tape=dr_tf_hts_tape,
            dx_tf_hts_tape_rebco=dx_tf_hts_tape_rebco,
            dx_tf_hts_tape_total=dx_tf_hts_tape_total,
            topology=self.topology,
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
