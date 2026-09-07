"""How big each coil has to be, and what it weighs."""

import equinox as eqx
import jax.numpy as jnp
from cottax.interfaces.pytree_namespace_module import (
    ExplicitFunction,
    From,
    FromExactly,
    OutputInto,
)

from functional_process.cottax.pfcoil import (
    CS_INDEX,
    N_PF_COILS,
    NGC2,
    REFERENCE_TOPOLOGY,
    SPHERICAL_TOKAMAK_TOPOLOGY,
    PFCoilTopology,
)
from functional_process.cottax.paths import fwbs, pf_coil, physics, tfcoil
from functional_process.models.pfcoil.masses import (
    calculate_pf_coil_masses,
    calculate_pf_coil_masses_no_central_solenoid,  # noqa: F401 -- re-exported for tests
    calculate_pf_coil_masses_no_central_solenoid_for_topology,
    calculate_pf_coil_sizes,
)

I_PF_SUPERCONDUCTOR = 3
"""`.pf_coil.i_pf_superconductor` on the reference run (`large_tokamak_eval.IN.DAT:246`)
-- NbTi, `pfcoil_variables.py:260`.
"""

I_CS_SUPERCONDUCTOR = 1
"""`.pf_coil.i_cs_superconductor` on the reference run (`:245`) -- ITER Nb3Sn,
`pfcoil_variables.py:256`.
"""

I_PF_SUPERCONDUCTOR_HAZELTON_ZHAI_REBCO = 9
"""`.pf_coil.i_pf_superconductor` on both spherical tokamaks
(`spherical_tokamak_eval.IN.DAT:235`, `st_regression.IN.DAT:1670`) -- Hazelton/Zhai
REBCO tape, `pfcoil_variables.py`'s `SuperconductorModel` value 9.
"""

I_CS_SUPERCONDUCTOR_WST_NB3SN = 5
"""`.pf_coil.i_cs_superconductor` on `low_aspect_ratio_DEMO.IN.DAT:845` -- WST Nb3Sn."""


class PFCoilSizes(ExplicitFunction):
    """cottax node: `.tokamak.pf_coil.sizes`."""

    topology: PFCoilTopology = eqx.field(static=True, default=REFERENCE_TOPOLOGY)
    """Static. Which slot each coil occupies and whether there is a CS slot to fill."""

    n_pf_coil_turns = OutputInto(pf_coil)
    r_pf_coil_inner = OutputInto(pf_coil)
    r_pf_coil_outer = OutputInto(pf_coil)
    z_pf_coil_upper = OutputInto(pf_coil)
    z_pf_coil_lower = OutputInto(pf_coil)
    r_pf_coil_outer_max = OutputInto(pf_coil)

    def __call__(
        self,
        c_pf_cs_coils_peak_ma=From(pf_coil),
        j_pf_coil_wp_peak=From(pf_coil),
        c_pf_coil_turn_peak_input=From(pf_coil),
        r_pf_coil_middle=From(pf_coil),
        z_pf_coil_middle=From(pf_coil),
        pf_current_safety_factor=From(pf_coil),
        r_cs_inner=From(pf_coil),
        r_cs_outer=From(pf_coil),
        z_cs_upper=From(pf_coil),
        z_cs_lower=From(pf_coil),
        rmajor=From(physics),
        rminor=From(physics),
        kappa=From(physics),
    ):
        return self._sized(
            c_pf_cs_coils_peak_ma,
            j_pf_coil_wp_peak,
            c_pf_coil_turn_peak_input,
            r_pf_coil_middle,
            z_pf_coil_middle,
            pf_current_safety_factor,
            r_cs_inner,
            r_cs_outer,
            z_cs_upper,
            z_cs_lower,
            rmajor,
            rminor,
            kappa,
        )

    def _sized(
        self,
        c_pf_cs_coils_peak_ma,
        j_pf_coil_wp_peak,
        c_pf_coil_turn_peak_input,
        r_pf_coil_middle,
        z_pf_coil_middle,
        pf_current_safety_factor,
        r_cs_inner,
        r_cs_outer,
        z_cs_upper,
        z_cs_lower,
        rmajor,
        rminor,
        kappa,
    ):
        """The sizing and its `NGC2` padding, given this arm's reads."""
        coils = self.topology.n_cs_pf_coils
        (
            turns,
            r_inner,
            r_outer,
            z_upper,
            z_lower,
            r_pf_coil_outer_max,
        ) = calculate_pf_coil_sizes(
            c_pf_cs_coils_peak_ma=c_pf_cs_coils_peak_ma[:coils],
            j_pf_coil_wp_peak=j_pf_coil_wp_peak[:coils],
            c_pf_coil_turn_peak_input=c_pf_coil_turn_peak_input[:coils],
            r_pf_coil_middle=r_pf_coil_middle[:coils],
            z_pf_coil_middle=z_pf_coil_middle[:coils],
            pf_current_safety_factor=pf_current_safety_factor,
            r_cs_inner=r_cs_inner,
            r_cs_outer=r_cs_outer,
            z_cs_upper=z_cs_upper,
            z_cs_lower=z_cs_lower,
            rmajor=rmajor,
            rminor=rminor,
            kappa=kappa,
            topology=self.topology,
        )
        pad = jnp.zeros(NGC2)
        filled = self.topology.plasma_index + 1
        return (
            pad.at[:filled].set(turns),
            pad.at[:filled].set(r_inner),
            pad.at[:filled].set(r_outer),
            pad.at[:filled].set(z_upper),
            pad.at[:filled].set(z_lower),
            r_pf_coil_outer_max,
        )


class PFCoilSizesNoCentralSolenoid(PFCoilSizes):
    """cottax node: `.tokamak.pf_coil.sizes`, the `iohcl = 0` occupant."""

    topology: PFCoilTopology = eqx.field(static=True, default=SPHERICAL_TOKAMAK_TOPOLOGY)

    def __call__(
        self,
        c_pf_cs_coils_peak_ma=From(pf_coil),
        j_pf_coil_wp_peak=From(pf_coil),
        c_pf_coil_turn_peak_input=From(pf_coil),
        r_pf_coil_middle=From(pf_coil),
        z_pf_coil_middle=From(pf_coil),
        pf_current_safety_factor=From(pf_coil),
        rmajor=From(physics),
        rminor=From(physics),
        kappa=From(physics),
    ):
        return self._sized(
            c_pf_cs_coils_peak_ma,
            j_pf_coil_wp_peak,
            c_pf_coil_turn_peak_input,
            r_pf_coil_middle,
            z_pf_coil_middle,
            pf_current_safety_factor,
            r_cs_inner=None,
            r_cs_outer=None,
            z_cs_upper=None,
            z_cs_lower=None,
            rmajor=rmajor,
            rminor=rminor,
            kappa=kappa,
        )


class PFCoilMasses(ExplicitFunction):
    """cottax node: `.tokamak.pf_coil.masses`."""

    m_pf_coil_conductor = OutputInto(pf_coil)
    m_pf_coil_structure = OutputInto(pf_coil)
    pfcaseth = OutputInto(pf_coil)
    m_pf_coil_conductor_total = OutputInto(pf_coil)
    m_pf_coil_structure_total = OutputInto(pf_coil)
    m_pf_coil_max = OutputInto(pf_coil)
    ricpf = OutputInto(pf_coil)
    a_cs_steel_poloidal = OutputInto(pf_coil)
    a_cs_cable_space = OutputInto(pf_coil)

    def _masses(
        self,
        c_pf_cs_coils_peak_ma,
        j_pf_coil_wp_peak,
        n_pf_coil_turns,
        r_pf_coil_middle,
        r_pf_coil_inner,
        r_pf_coil_outer,
        z_pf_coil_upper,
        z_pf_coil_lower,
        b_pf_coil_peak_0,
        b_pf_coil_peak_1,
        b_pf_coil_peak_2,
        b_pf_coil_peak_3,
        b_pf_coil_peak_4,
        b_pf_coil_peak_5,
        bpf2_0,
        bpf2_1,
        bpf2_2,
        bpf2_3,
        bpf2_4,
        bpf2_5,
        f_a_pf_coil_void,
        pf_current_safety_factor,
        sigpfcf,
        sigpfcalw,
        den_steel,
        den_pf_conductor,
        den_cs_conductor,
        a_cs_poloidal,
        f_a_cs_turn_steel,
        f_a_cs_void,
    ):
        """The whole calculation, given this occupant's two conductor densities."""
        b_peak = jnp.stack([
            b_pf_coil_peak_0,
            b_pf_coil_peak_1,
            b_pf_coil_peak_2,
            b_pf_coil_peak_3,
            b_pf_coil_peak_4,
            b_pf_coil_peak_5,
        ])
        b_outer = jnp.stack([bpf2_0, bpf2_1, bpf2_2, bpf2_3, bpf2_4, bpf2_5])

        (
            m_conductor,
            m_structure,
            pfcaseth,
            m_conductor_total,
            m_structure_total,
            m_pf_coil_max,
            ricpf,
            a_cs_steel_poloidal,
            a_cs_cable_space,
        ) = calculate_pf_coil_masses(
            c_pf_cs_coils_peak_ma=c_pf_cs_coils_peak_ma[: CS_INDEX + 1],
            j_pf_coil_wp_peak=j_pf_coil_wp_peak[: CS_INDEX + 1],
            n_pf_coil_turns=n_pf_coil_turns[: CS_INDEX + 1],
            r_pf_coil_middle=r_pf_coil_middle[: CS_INDEX + 1],
            r_pf_coil_inner=r_pf_coil_inner[: CS_INDEX + 1],
            r_pf_coil_outer=r_pf_coil_outer[: CS_INDEX + 1],
            z_pf_coil_upper=z_pf_coil_upper[: CS_INDEX + 1],
            z_pf_coil_lower=z_pf_coil_lower[: CS_INDEX + 1],
            b_pf_coil_peak=b_peak,
            bpf2=b_outer,
            f_a_pf_coil_void=f_a_pf_coil_void[:N_PF_COILS],
            pf_current_safety_factor=pf_current_safety_factor,
            sigpfcf=sigpfcf,
            sigpfcalw=sigpfcalw,
            den_steel=den_steel,
            den_pf_conductor=den_pf_conductor,
            den_cs_conductor=den_cs_conductor,
            a_cs_poloidal=a_cs_poloidal,
            f_a_cs_turn_steel=f_a_cs_turn_steel,
            f_a_cs_void=f_a_cs_void,
        )
        pad = jnp.zeros(NGC2)
        return (
            pad.at[: CS_INDEX + 1].set(m_conductor),
            pad.at[: CS_INDEX + 1].set(m_structure),
            pad.at[: CS_INDEX + 1].set(pfcaseth),
            m_conductor_total,
            m_structure_total,
            m_pf_coil_max,
            ricpf,
            a_cs_steel_poloidal,
            a_cs_cable_space,
        )

    def __call__(
        self,
        c_pf_cs_coils_peak_ma=From(pf_coil),
        j_pf_coil_wp_peak=From(pf_coil),
        n_pf_coil_turns=From(pf_coil),
        r_pf_coil_middle=From(pf_coil),
        r_pf_coil_inner=From(pf_coil),
        r_pf_coil_outer=From(pf_coil),
        z_pf_coil_upper=From(pf_coil),
        z_pf_coil_lower=From(pf_coil),
        b_pf_coil_peak_0=FromExactly(pf_coil.b_pf_coil_peak[0]),
        b_pf_coil_peak_1=FromExactly(pf_coil.b_pf_coil_peak[1]),
        b_pf_coil_peak_2=FromExactly(pf_coil.b_pf_coil_peak[2]),
        b_pf_coil_peak_3=FromExactly(pf_coil.b_pf_coil_peak[3]),
        b_pf_coil_peak_4=FromExactly(pf_coil.b_pf_coil_peak[4]),
        b_pf_coil_peak_5=FromExactly(pf_coil.b_pf_coil_peak[5]),
        bpf2_0=FromExactly(pf_coil.bpf2[0]),
        bpf2_1=FromExactly(pf_coil.bpf2[1]),
        bpf2_2=FromExactly(pf_coil.bpf2[2]),
        bpf2_3=FromExactly(pf_coil.bpf2[3]),
        bpf2_4=FromExactly(pf_coil.bpf2[4]),
        bpf2_5=FromExactly(pf_coil.bpf2[5]),
        f_a_pf_coil_void=From(pf_coil),
        pf_current_safety_factor=From(pf_coil),
        sigpfcf=From(pf_coil),
        sigpfcalw=From(pf_coil),
        den_steel=From(fwbs),
        den_pf_conductor=FromExactly(tfcoil.dcond[I_PF_SUPERCONDUCTOR - 1]),
        den_cs_conductor=FromExactly(tfcoil.dcond[I_CS_SUPERCONDUCTOR - 1]),
        a_cs_poloidal=From(pf_coil),
        f_a_cs_turn_steel=From(pf_coil),
        f_a_cs_void=From(pf_coil),
    ):
        return self._masses(
            c_pf_cs_coils_peak_ma,
            j_pf_coil_wp_peak,
            n_pf_coil_turns,
            r_pf_coil_middle,
            r_pf_coil_inner,
            r_pf_coil_outer,
            z_pf_coil_upper,
            z_pf_coil_lower,
            b_pf_coil_peak_0,
            b_pf_coil_peak_1,
            b_pf_coil_peak_2,
            b_pf_coil_peak_3,
            b_pf_coil_peak_4,
            b_pf_coil_peak_5,
            bpf2_0,
            bpf2_1,
            bpf2_2,
            bpf2_3,
            bpf2_4,
            bpf2_5,
            f_a_pf_coil_void,
            pf_current_safety_factor,
            sigpfcf,
            sigpfcalw,
            den_steel,
            den_pf_conductor,
            den_cs_conductor,
            a_cs_poloidal,
            f_a_cs_turn_steel,
            f_a_cs_void,
        )


class PFCoilMassesNoCentralSolenoid(ExplicitFunction):
    """cottax node: `.tokamak.pf_coil.masses`, the `iohcl = 0` occupant."""

    topology: PFCoilTopology = eqx.field(static=True, default=SPHERICAL_TOKAMAK_TOPOLOGY)

    m_pf_coil_conductor = OutputInto(pf_coil)
    m_pf_coil_structure = OutputInto(pf_coil)
    pfcaseth = OutputInto(pf_coil)
    m_pf_coil_conductor_total = OutputInto(pf_coil)
    m_pf_coil_structure_total = OutputInto(pf_coil)
    m_pf_coil_max = OutputInto(pf_coil)
    ricpf = OutputInto(pf_coil)

    def __call__(
        self,
        c_pf_cs_coils_peak_ma=From(pf_coil),
        j_pf_coil_wp_peak=From(pf_coil),
        n_pf_coil_turns=From(pf_coil),
        r_pf_coil_middle=From(pf_coil),
        r_pf_coil_inner=From(pf_coil),
        r_pf_coil_outer=From(pf_coil),
        z_pf_coil_upper=From(pf_coil),
        z_pf_coil_lower=From(pf_coil),
        b_pf_coil_peak=From(pf_coil),
        bpf2=From(pf_coil),
        f_a_pf_coil_void=From(pf_coil),
        pf_current_safety_factor=From(pf_coil),
        sigpfcf=From(pf_coil),
        sigpfcalw=From(pf_coil),
        den_steel=From(fwbs),
        den_pf_conductor=FromExactly(
            tfcoil.dcond[I_PF_SUPERCONDUCTOR_HAZELTON_ZHAI_REBCO - 1]
        ),
    ):
        return calculate_pf_coil_masses_no_central_solenoid_for_topology(
            c_pf_cs_coils_peak_ma=c_pf_cs_coils_peak_ma,
            j_pf_coil_wp_peak=j_pf_coil_wp_peak,
            n_pf_coil_turns=n_pf_coil_turns,
            r_pf_coil_middle=r_pf_coil_middle,
            r_pf_coil_inner=r_pf_coil_inner,
            r_pf_coil_outer=r_pf_coil_outer,
            z_pf_coil_upper=z_pf_coil_upper,
            z_pf_coil_lower=z_pf_coil_lower,
            b_pf_coil_peak=b_pf_coil_peak,
            bpf2=bpf2,
            f_a_pf_coil_void=f_a_pf_coil_void,
            pf_current_safety_factor=pf_current_safety_factor,
            sigpfcf=sigpfcf,
            sigpfcalw=sigpfcalw,
            den_steel=den_steel,
            den_pf_conductor=den_pf_conductor,
            topology=self.topology,
        )


class PFCoilMassesCsWstNb3Sn(PFCoilMasses):
    """cottax node: `.tokamak.pf_coil.masses`, the `i_cs_superconductor = 5` occupant.
    """

    def __call__(
        self,
        c_pf_cs_coils_peak_ma=From(pf_coil),
        j_pf_coil_wp_peak=From(pf_coil),
        n_pf_coil_turns=From(pf_coil),
        r_pf_coil_middle=From(pf_coil),
        r_pf_coil_inner=From(pf_coil),
        r_pf_coil_outer=From(pf_coil),
        z_pf_coil_upper=From(pf_coil),
        z_pf_coil_lower=From(pf_coil),
        b_pf_coil_peak_0=FromExactly(pf_coil.b_pf_coil_peak[0]),
        b_pf_coil_peak_1=FromExactly(pf_coil.b_pf_coil_peak[1]),
        b_pf_coil_peak_2=FromExactly(pf_coil.b_pf_coil_peak[2]),
        b_pf_coil_peak_3=FromExactly(pf_coil.b_pf_coil_peak[3]),
        b_pf_coil_peak_4=FromExactly(pf_coil.b_pf_coil_peak[4]),
        b_pf_coil_peak_5=FromExactly(pf_coil.b_pf_coil_peak[5]),
        bpf2_0=FromExactly(pf_coil.bpf2[0]),
        bpf2_1=FromExactly(pf_coil.bpf2[1]),
        bpf2_2=FromExactly(pf_coil.bpf2[2]),
        bpf2_3=FromExactly(pf_coil.bpf2[3]),
        bpf2_4=FromExactly(pf_coil.bpf2[4]),
        bpf2_5=FromExactly(pf_coil.bpf2[5]),
        f_a_pf_coil_void=From(pf_coil),
        pf_current_safety_factor=From(pf_coil),
        sigpfcf=From(pf_coil),
        sigpfcalw=From(pf_coil),
        den_steel=From(fwbs),
        den_pf_conductor=FromExactly(tfcoil.dcond[I_PF_SUPERCONDUCTOR - 1]),
        den_cs_conductor=FromExactly(tfcoil.dcond[I_CS_SUPERCONDUCTOR_WST_NB3SN - 1]),
        a_cs_poloidal=From(pf_coil),
        f_a_cs_turn_steel=From(pf_coil),
        f_a_cs_void=From(pf_coil),
    ):
        return self._masses(
            c_pf_cs_coils_peak_ma,
            j_pf_coil_wp_peak,
            n_pf_coil_turns,
            r_pf_coil_middle,
            r_pf_coil_inner,
            r_pf_coil_outer,
            z_pf_coil_upper,
            z_pf_coil_lower,
            b_pf_coil_peak_0,
            b_pf_coil_peak_1,
            b_pf_coil_peak_2,
            b_pf_coil_peak_3,
            b_pf_coil_peak_4,
            b_pf_coil_peak_5,
            bpf2_0,
            bpf2_1,
            bpf2_2,
            bpf2_3,
            bpf2_4,
            bpf2_5,
            f_a_pf_coil_void,
            pf_current_safety_factor,
            sigpfcf,
            sigpfcalw,
            den_steel,
            den_pf_conductor,
            den_cs_conductor,
            a_cs_poloidal,
            f_a_cs_turn_steel,
            f_a_cs_void,
        )
