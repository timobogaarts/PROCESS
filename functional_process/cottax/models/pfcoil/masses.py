"""How big each coil has to be, and what it weighs."""

import equinox as eqx
from cottax.interfaces.pytree_namespace_module import (
    ExplicitFunction,
    From,
    FromExactly,
    OutputInto,
)

from functional_process.cottax.models.pfcoil import (
    SPHERICAL_TOKAMAK_TOPOLOGY,
    PFCoilTopology,
)
from functional_process.cottax.paths import fwbs, pf_coil, physics, tfcoil
from functional_process.cottax.wraps import WrapsFunction
from functional_process.models.pfcoil.masses import (
    calculate_pf_coil_masses,  # noqa: F401 -- re-exported for tests
    calculate_pf_coil_masses_from_elements,
    calculate_pf_coil_masses_no_central_solenoid,  # noqa: F401 -- re-exported for tests
    calculate_pf_coil_masses_no_central_solenoid_for_topology,
    calculate_pf_coil_sizes,  # noqa: F401 -- re-exported for tests
    calculate_pf_coil_sizes_for_topology,
    calculate_pf_coil_sizes_from_elements,
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


class PFCoilSizes(WrapsFunction):
    """cottax node: `.tokamak.pf_coil.sizes`.

    Which slot each coil occupies and whether there is a CS slot to fill is fixed by
    `fn` (`calculate_pf_coil_sizes_from_elements`, for `REFERENCE_TOPOLOGY`) rather than
    carried as a static field here -- see `PFCoilSizesNoCentralSolenoid` for the other
    arm.
    """

    fn = calculate_pf_coil_sizes_from_elements

    c_pf_cs_coils_peak_ma = From(pf_coil)
    j_pf_coil_wp_peak = From(pf_coil)
    c_pf_coil_turn_peak_input = From(pf_coil)
    r_pf_coil_middle = From(pf_coil)
    z_pf_coil_middle = From(pf_coil)
    pf_current_safety_factor = From(pf_coil)
    r_cs_inner = From(pf_coil)
    r_cs_outer = From(pf_coil)
    z_cs_upper = From(pf_coil)
    z_cs_lower = From(pf_coil)
    rmajor = From(physics)
    rminor = From(physics)
    kappa = From(physics)

    n_pf_coil_turns = OutputInto(pf_coil)
    r_pf_coil_inner = OutputInto(pf_coil)
    r_pf_coil_outer = OutputInto(pf_coil)
    z_pf_coil_upper = OutputInto(pf_coil)
    z_pf_coil_lower = OutputInto(pf_coil)
    r_pf_coil_outer_max = OutputInto(pf_coil)


class PFCoilSizesNoCentralSolenoid(ExplicitFunction):
    """cottax node: `.tokamak.pf_coil.sizes`, the `iohcl = 0` occupant.

    Not `WrapsFunction`: `calculate_pf_coil_sizes_for_topology` takes a `topology`
    keyword this occupant must thread through explicitly (`self.topology`, see
    `PFCoilMassesNoCentralSolenoid` for the same shape elsewhere in this module).
    """

    topology: PFCoilTopology = eqx.field(static=True, default=SPHERICAL_TOKAMAK_TOPOLOGY)

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
        rmajor=From(physics),
        rminor=From(physics),
        kappa=From(physics),
    ):
        return calculate_pf_coil_sizes_for_topology(
            c_pf_cs_coils_peak_ma=c_pf_cs_coils_peak_ma,
            j_pf_coil_wp_peak=j_pf_coil_wp_peak,
            c_pf_coil_turn_peak_input=c_pf_coil_turn_peak_input,
            r_pf_coil_middle=r_pf_coil_middle,
            z_pf_coil_middle=z_pf_coil_middle,
            pf_current_safety_factor=pf_current_safety_factor,
            r_cs_inner=None,
            r_cs_outer=None,
            z_cs_upper=None,
            z_cs_lower=None,
            rmajor=rmajor,
            rminor=rminor,
            kappa=kappa,
            topology=self.topology,
        )


class PFCoilMasses(WrapsFunction):
    """cottax node: `.tokamak.pf_coil.masses`."""

    fn = calculate_pf_coil_masses_from_elements

    c_pf_cs_coils_peak_ma = From(pf_coil)
    j_pf_coil_wp_peak = From(pf_coil)
    n_pf_coil_turns = From(pf_coil)
    r_pf_coil_middle = From(pf_coil)
    r_pf_coil_inner = From(pf_coil)
    r_pf_coil_outer = From(pf_coil)
    z_pf_coil_upper = From(pf_coil)
    z_pf_coil_lower = From(pf_coil)
    b_pf_coil_peak_0 = FromExactly(pf_coil.b_pf_coil_peak[0])
    b_pf_coil_peak_1 = FromExactly(pf_coil.b_pf_coil_peak[1])
    b_pf_coil_peak_2 = FromExactly(pf_coil.b_pf_coil_peak[2])
    b_pf_coil_peak_3 = FromExactly(pf_coil.b_pf_coil_peak[3])
    b_pf_coil_peak_4 = FromExactly(pf_coil.b_pf_coil_peak[4])
    b_pf_coil_peak_5 = FromExactly(pf_coil.b_pf_coil_peak[5])
    bpf2_0 = FromExactly(pf_coil.bpf2[0])
    bpf2_1 = FromExactly(pf_coil.bpf2[1])
    bpf2_2 = FromExactly(pf_coil.bpf2[2])
    bpf2_3 = FromExactly(pf_coil.bpf2[3])
    bpf2_4 = FromExactly(pf_coil.bpf2[4])
    bpf2_5 = FromExactly(pf_coil.bpf2[5])
    f_a_pf_coil_void = From(pf_coil)
    pf_current_safety_factor = From(pf_coil)
    sigpfcf = From(pf_coil)
    sigpfcalw = From(pf_coil)
    den_steel = From(fwbs)
    den_pf_conductor = FromExactly(tfcoil.dcond[I_PF_SUPERCONDUCTOR - 1])
    den_cs_conductor = FromExactly(tfcoil.dcond[I_CS_SUPERCONDUCTOR - 1])
    a_cs_poloidal = From(pf_coil)
    f_a_cs_turn_steel = From(pf_coil)
    f_a_cs_void = From(pf_coil)

    m_pf_coil_conductor = OutputInto(pf_coil)
    m_pf_coil_structure = OutputInto(pf_coil)
    pfcaseth = OutputInto(pf_coil)
    m_pf_coil_conductor_total = OutputInto(pf_coil)
    m_pf_coil_structure_total = OutputInto(pf_coil)
    m_pf_coil_max = OutputInto(pf_coil)
    ricpf = OutputInto(pf_coil)
    a_cs_steel_poloidal = OutputInto(pf_coil)
    a_cs_cable_space = OutputInto(pf_coil)


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

    Shares every read and `fn` with `PFCoilMasses`; only the CS conductor density's
    source differs, so only that one attribute is redeclared.
    """

    den_cs_conductor = FromExactly(tfcoil.dcond[I_CS_SUPERCONDUCTOR_WST_NB3SN - 1])
