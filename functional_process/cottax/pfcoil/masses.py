"""How big each coil has to be, and what it weighs.

Audit record: `functional_process/_audit/units/models/pfcoil/masses.md`. This module
produces the three variables this wave's new consumers ask `.tokamak.pf_coil` for:
`.pf_coil.m_pf_coil_conductor_total` and `.pf_coil.m_pf_coil_structure_total` (read by
`models/structure.py::Structure`) and `.pf_coil.r_pf_coil_outer` (read by
`models/cryostat.py::Cryostat`).

Two units:

- `calculate_pf_coil_sizes` -- `pfcoil()`'s winding-pack geometry loop
  (`process/models/pfcoil.py:737-845`) for `i_pf_location != 1` coils, together with the
  CS's own slot (`:3237-3294`) and the plasma's (`:1067-1079`), so that each of the five
  per-coil arrays it produces has exactly one owner.
- `calculate_pf_coil_masses` -- the mass loop (`:849-1026`), the CS's steel and
  conductor (`:3504-3583`), and the summations (`:1028-1064`).

**What is deliberately not here.** The mass closure needs a coil's steel *area*, which
comes from the JxB hoop force and (for the CS) from `f_a_cs_turn_steel`. It does not
need any critical current, so `superconpf` (`:4641-4926`) and everything it reaches is
outside this closure and stays UNPORTED -- the ported superconductor fits in
`functional_process/models/physics/superconductors.py` are correspondingly *not* used
here, rather than being re-ported. Nor does it need the CS stress chain
(`calculate_cs_hoop_stress`, `calculate_cs_radial_stress`, the axial-stress profile and
`cs_fatigue.ncycle`, `:3403-3499`), which is the only part of `ohcalc` that touches
`scipy.special.ellipk`/`ellipe` and is therefore also the only part that would need a
new JAX primitive. Both exclusions are itemised in `masses.md`.

**The resistive arms are UNPORTED.** `i_pf_conductor = PFConductorModel.RESISTIVE`
changes `m_pf_coil_conductor`'s density, zeroes `areaspf` and `pfcaseth`, and adds
`p_pf_coil_resistive_total_flat_top`/`p_cs_resistive_flat_top`; that is a different
occupant, not a parameter of this one. On `large_tokamak_eval.IN.DAT`
`i_pf_conductor` takes its default `0` = `SUPERCONDUCTING`
(`pfcoil_variables.py:230`), which is the arm baked in below.
"""

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
"""`.pf_coil.i_pf_superconductor` on the reference run
(`large_tokamak_eval.IN.DAT:246`) -- NbTi, `pfcoil_variables.py:260`. It selects
`.tfcoil.dcond[2] = 6070` kg/m^3 and nothing else in this closure."""

I_CS_SUPERCONDUCTOR = 1
"""`.pf_coil.i_cs_superconductor` on the reference run (`:245`) -- ITER Nb3Sn,
`pfcoil_variables.py:256`. Selects `.tfcoil.dcond[0] = 6080` kg/m^3."""

I_PF_SUPERCONDUCTOR_HAZELTON_ZHAI_REBCO = 9
"""`.pf_coil.i_pf_superconductor` on both spherical tokamaks
(`spherical_tokamak_eval.IN.DAT:235`, `st_regression.IN.DAT:1670`) -- Hazelton/Zhai
REBCO tape, `pfcoil_variables.py`'s `SuperconductorModel` value 9. Selects
`.tfcoil.dcond[8]` as the PF conductor density and, in `superconpf`, the `hijc_rebco`
critical surface."""

I_CS_SUPERCONDUCTOR_WST_NB3SN = 5
"""`.pf_coil.i_cs_superconductor` on `low_aspect_ratio_DEMO.IN.DAT:845` -- WST Nb3Sn.
Selects `.tfcoil.dcond[4] = 6080` kg/m^3: numerically the same density as the ITER
Nb3Sn element, and still a different occupant, not a parameter -- an `i_*` integer may
not be a static kwarg even when two arms' arithmetic coincides (`masses.md`
§ switches touched, the `istore` precedent)."""


class PFCoilSizes(ExplicitFunction):
    """cottax node: `.tokamak.pf_coil.sizes`.

    Occupant for `i_pf_location != 1` on every group (`(2, 2, 3, 3)` here) with
    `iohcl = 1`. The `i_pf_location = 1` arm sizes a coil from the CS's radial thickness
    and the TF bore instead (`pfcoil.py:744-794`), and also *writes*
    `.pf_coil.j_pf_coil_wp_peak`, which this arm only reads -- a genuinely different
    read/write set, hence a different occupant. UNPORTED.

    Owns five per-coil arrays at their full `NGC2` width plus
    `.pf_coil.r_pf_coil_outer_max`. **One edge of this package's SCC**: it reads
    `.pf_coil.c_pf_cs_coils_peak_ma`, whose producer chain runs back through
    `currents.CSFluxSwing`, which reads `.pf_coil.n_pf_coil_turns` from here. See
    `currents.py`'s module docstring.
    """

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
    """cottax node: `.tokamak.pf_coil.sizes`, the `iohcl = 0` occupant.

    **Four reads fewer**: the CS's own four edges (`r_cs_inner`, `r_cs_outer`,
    `z_cs_upper`, `z_cs_lower`) fill index `n_cs_pf_coils - 1` of the four edge arrays
    on the conventional arm, and with no solenoid there is no such index -- `ohcalc`,
    which writes them, is never entered (`pfcoil.py:1048-1050`). The plasma still gets
    its slot (`:1067-1079`), one index further along than on a machine with a CS.
    """

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
    """cottax node: `.tokamak.pf_coil.masses`.

    Occupant for `i_pf_conductor = SUPERCONDUCTING`, `i_pf_superconductor = 3` (NbTi)
    and `i_cs_superconductor = 1` (ITER Nb3Sn) with `iohcl = 1`. The two superconductor
    switches enter only as the index of a density in `.tfcoil.dcond`, so they are
    `FromExactly`s at a fixed index -- the same shape
    `models/tfcoil/superconducting.py:1499` and
    `models/stellarator/coils/mass.py:224` already use -- and a different material is a
    different occupant, not a different argument.

    Owns `.pf_coil.m_pf_coil_conductor`, `.pf_coil.m_pf_coil_structure` and
    `.pf_coil.pfcaseth` whole, plus the six scalars derived from them. The six per-index
    reads of `.pf_coil.b_pf_coil_peak`/`.pf_coil.bpf2` match `fields.PFCoilPeakField`'s
    per-index `Output`s: index 6 of both arrays belongs to the CS's own self-field, which
    is UNPORTED and which no mass here depends on.

    `PFCoilMassesCsWstNb3Sn` below is the occupant family's second member -- the
    `CoilsMass` shape (`models/stellarator/coils/mass.py`): the whole calculation lives
    in `_masses`, each occupant's `__call__` declares its own ports
    (`ExplicitFunction._signature_of` reads `__call__`'s signature only), and the only
    entry that differs between them is which `.tfcoil.dcond` element is the CS
    conductor density.
    """

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
        """The whole calculation, given this occupant's two conductor densities.

        Not a port surface: `_params` reads `__call__`'s signature only, so what each
        occupant declares is still its own parameter list -- and the only entry that
        differs between them is the `.tfcoil.dcond[k]` element bound to
        `den_cs_conductor`.
        """
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
    """cottax node: `.tokamak.pf_coil.masses`, the `iohcl = 0` occupant.

    Not a subclass of `PFCoilMasses`: **it owns two outputs fewer**, and an `Output` is
    inherited by attribute name (`_declared_outputs_on_cls`), so a subclass could add
    slots but not drop them. `.pf_coil.a_cs_steel_poloidal` and
    `.pf_coil.a_cs_cable_space` come from `ohcalc` (`pfcoil.py:3504-3583`), which
    `pfcoil()` skips entirely when there is no solenoid (`:1048-1050`) -- so on this
    machine they have no producer, which is the honest answer and not a pair of zeros.
    `den_cs_conductor` disappears with them, and with it the whole
    `(i_pf_superconductor, i_cs_superconductor)` pair's *second* half: the CS
    superconductor switch has nothing left to select.

    `.pf_coil.b_pf_coil_peak` and `.pf_coil.bpf2` are read **whole** here rather than as
    six per-index reads, matching `PFCoilPeakFieldNoCentralSolenoid`, which owns them
    whole for the mirror-image reason.

    Occupant for `i_pf_conductor = SUPERCONDUCTING` with
    `i_pf_superconductor = HAZELTON_ZHAI_REBCO` (9) --
    `spherical_tokamak_eval.IN.DAT:235` and `st_regression.IN.DAT:1670`. The switch
    enters this node only as the index of a density in `.tfcoil.dcond`, the same shape
    `PFCoilMasses` uses; a different material is a different occupant.
    """

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

    Occupant for `i_pf_conductor = SUPERCONDUCTING`, `i_pf_superconductor = 3` (NbTi)
    and `i_cs_superconductor = 5` (WST Nb3Sn) with `iohcl = 1` --
    `low_aspect_ratio_DEMO.IN.DAT`'s pair (`:806`, `:845`). Differs from `PFCoilMasses`
    in exactly one binding: the CS conductor density is `.tfcoil.dcond[4]` instead of
    `.tfcoil.dcond[0]`. The two elements hold the same 6080 kg/m^3 today, and the split
    is still the point: the read moves *with* the switch, where a baked `dcond[0]`
    would silently keep reading ITER Nb3Sn's slot on a WST machine -- the `CoilsMass`
    lesson, `_audit/next_steps.md` §14.11.
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
