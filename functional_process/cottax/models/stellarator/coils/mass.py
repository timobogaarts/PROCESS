"""Pure-functional port of `process/models/stellarator/coils/mass.py` (registry #12)."""

from cottax.interfaces.pytree_namespace_module import (
    ExplicitFunction,
    From,
    FromExactly,
    OutputInto,
)

from functional_process.cottax.paths import (
    fwbs,
    tfcoil,
)
from functional_process.models.stellarator.coils.mass import (
    calculate_coils_mass,
)
from functional_process.vocabulary import (
    constants,  # noqa: F401
)


class CoilsMass(ExplicitFunction):
    """The `calculate_coils_mass` family -- one occupant per `.tfcoil.i_tf_sc_mat`
    value, differing only in which element of `.tfcoil.dcond` is the superconductor
    density.
    """

    m_tf_coil_case = OutputInto(tfcoil)
    m_tf_coil_wp_insulation = OutputInto(tfcoil)
    m_tf_coil_superconductor = OutputInto(tfcoil)
    m_tf_coil_copper = OutputInto(tfcoil)
    m_tf_wp_steel_conduit = OutputInto(tfcoil)
    m_tf_coil_wp_turn_insulation = OutputInto(tfcoil)
    m_tf_coil_conductor = OutputInto(tfcoil)
    m_tf_coils_total = OutputInto(tfcoil)

    def _masses(
        self,
        den_tf_sc_material,
        a_tf_wp_with_insulation,
        a_tf_wp_no_insulation,
        len_tf_coil,
        a_tf_coil_inboard_case,
        den_tf_coil_case,
        den_tf_wp_turn_insulation,
        n_tf_coil_turns,
        a_tf_turn_cable_space_no_void,
        f_a_tf_turn_cable_space_extra_void,
        f_a_tf_turn_cable_copper,
        a_tf_wp_coolant_channels,
        a_tf_turn_steel,
        a_tf_coil_wp_turn_insulation,
        n_tf_coils,
        den_steel,
    ):
        """The whole calculation, given this material's density."""
        return calculate_coils_mass(
            a_tf_wp_with_insulation,
            a_tf_wp_no_insulation,
            len_tf_coil,
            a_tf_coil_inboard_case,
            den_tf_coil_case,
            den_tf_wp_turn_insulation,
            n_tf_coil_turns,
            a_tf_turn_cable_space_no_void,
            f_a_tf_turn_cable_space_extra_void,
            f_a_tf_turn_cable_copper,
            a_tf_wp_coolant_channels,
            den_tf_sc_material,
            a_tf_turn_steel,
            den_steel,
            a_tf_coil_wp_turn_insulation,
            n_tf_coils,
        )


class IterNb3snCoilsMass(CoilsMass):
    """`i_tf_sc_mat == ITER_NB3SN` (1) -- ITER Nb3Sn -- PROCESS's own default and the
    reference run's.
    """

    def __call__(
        self,
        den_tf_sc_material=FromExactly(tfcoil.dcond[0]),
        a_tf_wp_with_insulation=From(tfcoil),
        a_tf_wp_no_insulation=From(tfcoil),
        len_tf_coil=From(tfcoil),
        a_tf_coil_inboard_case=From(tfcoil),
        den_tf_coil_case=From(tfcoil),
        den_tf_wp_turn_insulation=From(tfcoil),
        n_tf_coil_turns=From(tfcoil),
        a_tf_turn_cable_space_no_void=From(tfcoil),
        f_a_tf_turn_cable_space_extra_void=From(tfcoil),
        f_a_tf_turn_cable_copper=From(tfcoil),
        a_tf_wp_coolant_channels=From(tfcoil),
        a_tf_turn_steel=From(tfcoil),
        a_tf_coil_wp_turn_insulation=From(tfcoil),
        n_tf_coils=From(tfcoil),
        den_steel=From(fwbs),
    ):
        return self._masses(
            den_tf_sc_material,
            a_tf_wp_with_insulation,
            a_tf_wp_no_insulation,
            len_tf_coil,
            a_tf_coil_inboard_case,
            den_tf_coil_case,
            den_tf_wp_turn_insulation,
            n_tf_coil_turns,
            a_tf_turn_cable_space_no_void,
            f_a_tf_turn_cable_space_extra_void,
            f_a_tf_turn_cable_copper,
            a_tf_wp_coolant_channels,
            a_tf_turn_steel,
            a_tf_coil_wp_turn_insulation,
            n_tf_coils,
            den_steel,
        )


class Bi2212CoilsMass(CoilsMass):
    """`i_tf_sc_mat == BI2212` (2) -- Bi-2212."""

    def __call__(
        self,
        den_tf_sc_material=FromExactly(tfcoil.dcond[1]),
        a_tf_wp_with_insulation=From(tfcoil),
        a_tf_wp_no_insulation=From(tfcoil),
        len_tf_coil=From(tfcoil),
        a_tf_coil_inboard_case=From(tfcoil),
        den_tf_coil_case=From(tfcoil),
        den_tf_wp_turn_insulation=From(tfcoil),
        n_tf_coil_turns=From(tfcoil),
        a_tf_turn_cable_space_no_void=From(tfcoil),
        f_a_tf_turn_cable_space_extra_void=From(tfcoil),
        f_a_tf_turn_cable_copper=From(tfcoil),
        a_tf_wp_coolant_channels=From(tfcoil),
        a_tf_turn_steel=From(tfcoil),
        a_tf_coil_wp_turn_insulation=From(tfcoil),
        n_tf_coils=From(tfcoil),
        den_steel=From(fwbs),
    ):
        return self._masses(
            den_tf_sc_material,
            a_tf_wp_with_insulation,
            a_tf_wp_no_insulation,
            len_tf_coil,
            a_tf_coil_inboard_case,
            den_tf_coil_case,
            den_tf_wp_turn_insulation,
            n_tf_coil_turns,
            a_tf_turn_cable_space_no_void,
            f_a_tf_turn_cable_space_extra_void,
            f_a_tf_turn_cable_copper,
            a_tf_wp_coolant_channels,
            a_tf_turn_steel,
            a_tf_coil_wp_turn_insulation,
            n_tf_coils,
            den_steel,
        )


class OldLubellNbtiCoilsMass(CoilsMass):
    """`i_tf_sc_mat == OLD_LUBELL_NBTI` (3) -- old Lubell NbTi."""

    def __call__(
        self,
        den_tf_sc_material=FromExactly(tfcoil.dcond[2]),
        a_tf_wp_with_insulation=From(tfcoil),
        a_tf_wp_no_insulation=From(tfcoil),
        len_tf_coil=From(tfcoil),
        a_tf_coil_inboard_case=From(tfcoil),
        den_tf_coil_case=From(tfcoil),
        den_tf_wp_turn_insulation=From(tfcoil),
        n_tf_coil_turns=From(tfcoil),
        a_tf_turn_cable_space_no_void=From(tfcoil),
        f_a_tf_turn_cable_space_extra_void=From(tfcoil),
        f_a_tf_turn_cable_copper=From(tfcoil),
        a_tf_wp_coolant_channels=From(tfcoil),
        a_tf_turn_steel=From(tfcoil),
        a_tf_coil_wp_turn_insulation=From(tfcoil),
        n_tf_coils=From(tfcoil),
        den_steel=From(fwbs),
    ):
        return self._masses(
            den_tf_sc_material,
            a_tf_wp_with_insulation,
            a_tf_wp_no_insulation,
            len_tf_coil,
            a_tf_coil_inboard_case,
            den_tf_coil_case,
            den_tf_wp_turn_insulation,
            n_tf_coil_turns,
            a_tf_turn_cable_space_no_void,
            f_a_tf_turn_cable_space_extra_void,
            f_a_tf_turn_cable_copper,
            a_tf_wp_coolant_channels,
            a_tf_turn_steel,
            a_tf_coil_wp_turn_insulation,
            n_tf_coils,
            den_steel,
        )


class UserDefinedNb3snCoilsMass(CoilsMass):
    """`i_tf_sc_mat == USER_DEFINED_NB3SN` (4) -- user-defined ITER Nb3Sn."""

    def __call__(
        self,
        den_tf_sc_material=FromExactly(tfcoil.dcond[3]),
        a_tf_wp_with_insulation=From(tfcoil),
        a_tf_wp_no_insulation=From(tfcoil),
        len_tf_coil=From(tfcoil),
        a_tf_coil_inboard_case=From(tfcoil),
        den_tf_coil_case=From(tfcoil),
        den_tf_wp_turn_insulation=From(tfcoil),
        n_tf_coil_turns=From(tfcoil),
        a_tf_turn_cable_space_no_void=From(tfcoil),
        f_a_tf_turn_cable_space_extra_void=From(tfcoil),
        f_a_tf_turn_cable_copper=From(tfcoil),
        a_tf_wp_coolant_channels=From(tfcoil),
        a_tf_turn_steel=From(tfcoil),
        a_tf_coil_wp_turn_insulation=From(tfcoil),
        n_tf_coils=From(tfcoil),
        den_steel=From(fwbs),
    ):
        return self._masses(
            den_tf_sc_material,
            a_tf_wp_with_insulation,
            a_tf_wp_no_insulation,
            len_tf_coil,
            a_tf_coil_inboard_case,
            den_tf_coil_case,
            den_tf_wp_turn_insulation,
            n_tf_coil_turns,
            a_tf_turn_cable_space_no_void,
            f_a_tf_turn_cable_space_extra_void,
            f_a_tf_turn_cable_copper,
            a_tf_wp_coolant_channels,
            a_tf_turn_steel,
            a_tf_coil_wp_turn_insulation,
            n_tf_coils,
            den_steel,
        )


class WstNb3snCoilsMass(CoilsMass):
    """`i_tf_sc_mat == WST_NB3SN` (5) -- WST Nb3Sn."""

    def __call__(
        self,
        den_tf_sc_material=FromExactly(tfcoil.dcond[4]),
        a_tf_wp_with_insulation=From(tfcoil),
        a_tf_wp_no_insulation=From(tfcoil),
        len_tf_coil=From(tfcoil),
        a_tf_coil_inboard_case=From(tfcoil),
        den_tf_coil_case=From(tfcoil),
        den_tf_wp_turn_insulation=From(tfcoil),
        n_tf_coil_turns=From(tfcoil),
        a_tf_turn_cable_space_no_void=From(tfcoil),
        f_a_tf_turn_cable_space_extra_void=From(tfcoil),
        f_a_tf_turn_cable_copper=From(tfcoil),
        a_tf_wp_coolant_channels=From(tfcoil),
        a_tf_turn_steel=From(tfcoil),
        a_tf_coil_wp_turn_insulation=From(tfcoil),
        n_tf_coils=From(tfcoil),
        den_steel=From(fwbs),
    ):
        return self._masses(
            den_tf_sc_material,
            a_tf_wp_with_insulation,
            a_tf_wp_no_insulation,
            len_tf_coil,
            a_tf_coil_inboard_case,
            den_tf_coil_case,
            den_tf_wp_turn_insulation,
            n_tf_coil_turns,
            a_tf_turn_cable_space_no_void,
            f_a_tf_turn_cable_space_extra_void,
            f_a_tf_turn_cable_copper,
            a_tf_wp_coolant_channels,
            a_tf_turn_steel,
            a_tf_coil_wp_turn_insulation,
            n_tf_coils,
            den_steel,
        )


class CrocoRebcoCoilsMass(CoilsMass):
    """`i_tf_sc_mat == CROCO_REBCO` (6) -- CroCo REBCO."""

    def __call__(
        self,
        den_tf_sc_material=FromExactly(tfcoil.dcond[5]),
        a_tf_wp_with_insulation=From(tfcoil),
        a_tf_wp_no_insulation=From(tfcoil),
        len_tf_coil=From(tfcoil),
        a_tf_coil_inboard_case=From(tfcoil),
        den_tf_coil_case=From(tfcoil),
        den_tf_wp_turn_insulation=From(tfcoil),
        n_tf_coil_turns=From(tfcoil),
        a_tf_turn_cable_space_no_void=From(tfcoil),
        f_a_tf_turn_cable_space_extra_void=From(tfcoil),
        f_a_tf_turn_cable_copper=From(tfcoil),
        a_tf_wp_coolant_channels=From(tfcoil),
        a_tf_turn_steel=From(tfcoil),
        a_tf_coil_wp_turn_insulation=From(tfcoil),
        n_tf_coils=From(tfcoil),
        den_steel=From(fwbs),
    ):
        return self._masses(
            den_tf_sc_material,
            a_tf_wp_with_insulation,
            a_tf_wp_no_insulation,
            len_tf_coil,
            a_tf_coil_inboard_case,
            den_tf_coil_case,
            den_tf_wp_turn_insulation,
            n_tf_coil_turns,
            a_tf_turn_cable_space_no_void,
            f_a_tf_turn_cable_space_extra_void,
            f_a_tf_turn_cable_copper,
            a_tf_wp_coolant_channels,
            a_tf_turn_steel,
            a_tf_coil_wp_turn_insulation,
            n_tf_coils,
            den_steel,
        )


class DurhamNbtiCoilsMass(CoilsMass):
    """`i_tf_sc_mat == DURHAM_NBTI` (7) -- Durham Ginzburg-Landau NbTi."""

    def __call__(
        self,
        den_tf_sc_material=FromExactly(tfcoil.dcond[6]),
        a_tf_wp_with_insulation=From(tfcoil),
        a_tf_wp_no_insulation=From(tfcoil),
        len_tf_coil=From(tfcoil),
        a_tf_coil_inboard_case=From(tfcoil),
        den_tf_coil_case=From(tfcoil),
        den_tf_wp_turn_insulation=From(tfcoil),
        n_tf_coil_turns=From(tfcoil),
        a_tf_turn_cable_space_no_void=From(tfcoil),
        f_a_tf_turn_cable_space_extra_void=From(tfcoil),
        f_a_tf_turn_cable_copper=From(tfcoil),
        a_tf_wp_coolant_channels=From(tfcoil),
        a_tf_turn_steel=From(tfcoil),
        a_tf_coil_wp_turn_insulation=From(tfcoil),
        n_tf_coils=From(tfcoil),
        den_steel=From(fwbs),
    ):
        return self._masses(
            den_tf_sc_material,
            a_tf_wp_with_insulation,
            a_tf_wp_no_insulation,
            len_tf_coil,
            a_tf_coil_inboard_case,
            den_tf_coil_case,
            den_tf_wp_turn_insulation,
            n_tf_coil_turns,
            a_tf_turn_cable_space_no_void,
            f_a_tf_turn_cable_space_extra_void,
            f_a_tf_turn_cable_copper,
            a_tf_wp_coolant_channels,
            a_tf_turn_steel,
            a_tf_coil_wp_turn_insulation,
            n_tf_coils,
            den_steel,
        )


class DurhamRebcoCoilsMass(CoilsMass):
    """`i_tf_sc_mat == DURHAM_REBCO` (8) -- Durham REBCO."""

    def __call__(
        self,
        den_tf_sc_material=FromExactly(tfcoil.dcond[7]),
        a_tf_wp_with_insulation=From(tfcoil),
        a_tf_wp_no_insulation=From(tfcoil),
        len_tf_coil=From(tfcoil),
        a_tf_coil_inboard_case=From(tfcoil),
        den_tf_coil_case=From(tfcoil),
        den_tf_wp_turn_insulation=From(tfcoil),
        n_tf_coil_turns=From(tfcoil),
        a_tf_turn_cable_space_no_void=From(tfcoil),
        f_a_tf_turn_cable_space_extra_void=From(tfcoil),
        f_a_tf_turn_cable_copper=From(tfcoil),
        a_tf_wp_coolant_channels=From(tfcoil),
        a_tf_turn_steel=From(tfcoil),
        a_tf_coil_wp_turn_insulation=From(tfcoil),
        n_tf_coils=From(tfcoil),
        den_steel=From(fwbs),
    ):
        return self._masses(
            den_tf_sc_material,
            a_tf_wp_with_insulation,
            a_tf_wp_no_insulation,
            len_tf_coil,
            a_tf_coil_inboard_case,
            den_tf_coil_case,
            den_tf_wp_turn_insulation,
            n_tf_coil_turns,
            a_tf_turn_cable_space_no_void,
            f_a_tf_turn_cable_space_extra_void,
            f_a_tf_turn_cable_copper,
            a_tf_wp_coolant_channels,
            a_tf_turn_steel,
            a_tf_coil_wp_turn_insulation,
            n_tf_coils,
            den_steel,
        )
