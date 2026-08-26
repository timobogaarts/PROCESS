"""Pure-functional port of `process/models/stellarator/coils/mass.py` (registry #12).

Audit record: `functional_process/_audit/units/models/stellarator/coils/mass.md`. The
source's `calculate_coils_mass` orchestrates 8 sub-functions (`casing`,
`ground_insulation`, `superconductor`, `copper`, `conduit_steel`, `conduit_insulation`,
`total_conductor`, `total_coil`), each writing one `data.tfcoil.*` field that a later
sub-function reads straight back off `data` -- unconditional, unbranched, same-call
produce-then-consume, so this is `local-intermediate` exactly like
`structure.md`'s `aintmass` chain, just one file over. Ported here as one
straight-line function with ordinary Python locals instead of eight `data`-mediated
steps.

`superconductor()`'s `data.tfcoil.dcond[data.tfcoil.i_tf_sc_mat - 1]`
(`process/models/stellarator/coils/mass.py:88`) is a data-table lookup (material
density), not a formula branch. The *pure function* below still takes it as one
already-indexed scalar argument, `den_tf_sc_material` -- that part is unchanged.

**What changed** (MDA triage, `_audit/next_steps.md` §8.1, row
`.tfcoil.den_tf_sc_material`): **`CoilsMass`'s `FromExactly` no longer mints
`.tfcoil.den_tf_sc_material`.** `dcond` is a real
`DataStructure` field (`process/data_structure/tfcoil_variables.py:157-170`, nine fixed
material densities), so the read has a real `VarPath` and does not need one invented:
`.tfcoil.dcond[i_tf_sc_mat - 1]`, an array-element path exactly as
`_audit/naming_convention.md` § "Array elements" prescribes and as
`physics/radiation_power.py:619-660` already binds
`.impurity_radiation.f_nd_impurity_electron_array[0..13]`. This also answers the record's
own open question 1 ("should whoever designs the `i_tf_sc_mat` node mint its output under
this exact name") in the negative: no lookup node is needed at all, because the lookup's
*input* is already a real place and its index is static.

`i_tf_sc_mat` is a topology switch, so per `_audit/naming_convention.md` § "switches are
not ports" it is resolved when the graph is assembled, not carried as a port -- and a
`FromExactly` default is fixed at class-definition time, so the index it selects is fixed
with it. **That made `CoilsMass` a family, and until `_audit/next_steps.md` §14.2 it was
one class pretending not to be**: a module constant `I_TF_SC_MAT_ITER_NB3SN = 1` chose
`dcond[0]`, invisible to `switch_audit` (which walks `eqx.field(static=True)` and nothing
else) and to every other switch instrument in the port. There are eight occupants now,
keyed by `indat.COILS_MASS_MATERIAL` on the same value
`indat.WINDING_PACK_MATERIAL` uses, so the coil-mass node and the winding-pack node
cannot name different materials.
"""

from cottax.interfaces.pytree_namespace_module import (
    ExplicitFunction,
    From,
    FromExactly,
    OutputInto,
)

from functional_process.paths import fwbs, tfcoil
from process.core import constants


def calculate_coils_mass(
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
):
    """Total coil mass, and the intermediate component masses that feed it.

    Ports `calculate_coils_mass`'s 8-step chain (see module docstring).

    Parameters
    ----------
    a_tf_wp_with_insulation, a_tf_wp_no_insulation :
        Winding pack area, with/without insulation (m2).
    len_tf_coil :
        TF coil length (m). `.tfcoil.len_tf_coil`.
    a_tf_coil_inboard_case :
        TF coil case area (m2). `.tfcoil.a_tf_coil_inboard_case`.
    den_tf_coil_case :
        Case material density (kg/m3). `.tfcoil.den_tf_coil_case`.
    den_tf_wp_turn_insulation :
        Turn/ground insulation density (kg/m3). `.tfcoil.den_tf_wp_turn_insulation`.
    n_tf_coil_turns :
        Turns per coil. `.tfcoil.n_tf_coil_turns`.
    a_tf_turn_cable_space_no_void :
        Cable space area per turn, no void (m2). `.tfcoil.a_tf_turn_cable_space_no_void`.
    f_a_tf_turn_cable_space_extra_void :
        Extra-void fraction of cable space. `.tfcoil.f_a_tf_turn_cable_space_extra_void`.
    f_a_tf_turn_cable_copper :
        Copper fraction of cable conductor. `.tfcoil.f_a_tf_turn_cable_copper`.
    a_tf_wp_coolant_channels :
        Coolant channel area (m2, 0 for a stellarator). `.tfcoil.a_tf_wp_coolant_channels`.
    den_tf_sc_material :
        Superconductor density (kg/m3) -- `.tfcoil.dcond[i_tf_sc_mat - 1]`, already
        indexed by the material switch (see module docstring).
    a_tf_turn_steel :
        Steel conduit area per turn (m2). `.tfcoil.a_tf_turn_steel`.
    den_steel :
        Steel density (kg/m3). `.fwbs.den_steel`.
    a_tf_coil_wp_turn_insulation :
        Turn insulation area, already `n_tf_coil_turns`-scaled (m2).
        `.tfcoil.a_tf_coil_wp_turn_insulation`.
    n_tf_coils :
        Number of TF coils. `.tfcoil.n_tf_coils`.

    Returns
    -------
    :
        `(m_tf_coil_case, m_tf_coil_wp_insulation, m_tf_coil_superconductor,
        m_tf_coil_copper, m_tf_wp_steel_conduit, m_tf_coil_wp_turn_insulation,
        m_tf_coil_conductor, m_tf_coils_total)` -- all masses (kg).
    """
    m_tf_coil_case = len_tf_coil * a_tf_coil_inboard_case * den_tf_coil_case

    m_tf_coil_wp_insulation = (
        len_tf_coil
        * (a_tf_wp_with_insulation - a_tf_wp_no_insulation)
        * den_tf_wp_turn_insulation
    )

    m_tf_coil_superconductor = (
        len_tf_coil
        * n_tf_coil_turns
        * a_tf_turn_cable_space_no_void
        * (1.0 - f_a_tf_turn_cable_space_extra_void)
        * (1.0 - f_a_tf_turn_cable_copper)
        - len_tf_coil * a_tf_wp_coolant_channels
    ) * den_tf_sc_material

    m_tf_coil_copper = (
        len_tf_coil
        * n_tf_coil_turns
        * a_tf_turn_cable_space_no_void
        * (1.0 - f_a_tf_turn_cable_space_extra_void)
        * f_a_tf_turn_cable_copper
        - len_tf_coil * a_tf_wp_coolant_channels
    ) * constants.DEN_COPPER

    m_tf_wp_steel_conduit = len_tf_coil * n_tf_coil_turns * a_tf_turn_steel * den_steel

    m_tf_coil_wp_turn_insulation = (
        len_tf_coil * a_tf_coil_wp_turn_insulation * den_tf_wp_turn_insulation
    )

    m_tf_coil_conductor = (
        m_tf_coil_superconductor
        + m_tf_coil_copper
        + m_tf_wp_steel_conduit
        + m_tf_coil_wp_turn_insulation
    )

    m_tf_coils_total = (
        m_tf_coil_case + m_tf_coil_conductor + m_tf_coil_wp_insulation
    ) * n_tf_coils

    return (
        m_tf_coil_case,
        m_tf_coil_wp_insulation,
        m_tf_coil_superconductor,
        m_tf_coil_copper,
        m_tf_wp_steel_conduit,
        m_tf_coil_wp_turn_insulation,
        m_tf_coil_conductor,
        m_tf_coils_total,
    )


class CoilsMass(ExplicitFunction):
    """The `calculate_coils_mass` family -- one occupant per `.tfcoil.i_tf_sc_mat`
    value, differing only in which element of `.tfcoil.dcond` is the superconductor
    density.

    **`i_tf_sc_mat` was answered here by a module constant, not by a static kwarg, and
    that is why nothing caught it** (`_audit/next_steps.md` §14.2). `switch_audit` walks
    `eqx.field(static=True)` attributes on the assembled graph; a module-level
    `I_TF_SC_MAT_ITER_NB3SN = 1` baked into a `FromExactly` default is invisible to it,
    to `test_switch_coverage.test_no_slot_contradicts_a_factory_switch`, and to every
    other instrument the port has. The node one slot over --
    `stellarator.coils.winding_pack_intersect_inputs` -- has been a real
    eight-occupant family since §14.5, so an `i_tf_sc_mat = 5` machine assembled
    `WstNb3snWindingPackIntersectInputs` next to a coil-mass node still reading
    `dcond[0]`: the same incoherence band (a) of `switch_kwarg_survey.md` found five
    times over, one layer below where anything was looking.

    `dcond` is a real `DataStructure` field (`tfcoil_variables.py:157-170`, nine fixed
    material densities), so each occupant binds a real array-element `VarPath` --
    `_audit/naming_convention.md` § "Array elements" -- and no lookup node is minted.
    The eight occupants are keyed by `indat.COILS_MASS_MATERIAL`, which is
    `WINDING_PACK_MATERIAL`'s own key, so the two nodes cannot disagree by
    construction.

    Its two winding-pack area `From`s (`.tfcoil.a_tf_wp_with_insulation`/
    `.a_tf_wp_no_insulation`) *are* minted, but by this port's own
    `coils/calculate.py` (`WindingPackTotalSizePost`), which is their producer; they are
    locals in PROCESS (`process/models/stellarator/coils/calculate.py:496-501`, the
    source's own comment: "not global").
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
        """The whole calculation, given this material's density.

        Not a port surface: `_params` reads `__call__`'s signature only
        (`ExplicitFunction._signature_of`), so what each occupant declares is still its
        own parameter list -- and the only entry that differs between them is the
        `.tfcoil.dcond[k]` element.
        """
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
    """`i_tf_sc_mat == ITER_NB3SN` (1) -- ITER Nb3Sn -- PROCESS's own default and the reference run's.

    Reads `.tfcoil.dcond[0]` as the superconductor density.
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
    """`i_tf_sc_mat == BI2212` (2) -- Bi-2212.

    Reads `.tfcoil.dcond[1]` as the superconductor density.
    """

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
    """`i_tf_sc_mat == OLD_LUBELL_NBTI` (3) -- old Lubell NbTi.

    Reads `.tfcoil.dcond[2]` as the superconductor density.
    """

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
    """`i_tf_sc_mat == USER_DEFINED_NB3SN` (4) -- user-defined ITER Nb3Sn.

    Reads `.tfcoil.dcond[3]` as the superconductor density.
    """

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
    """`i_tf_sc_mat == WST_NB3SN` (5) -- WST Nb3Sn.

    Reads `.tfcoil.dcond[4]` as the superconductor density.
    """

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
    """`i_tf_sc_mat == CROCO_REBCO` (6) -- CroCo REBCO.

    Reads `.tfcoil.dcond[5]` as the superconductor density.
    """

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
    """`i_tf_sc_mat == DURHAM_NBTI` (7) -- Durham Ginzburg-Landau NbTi.

    Reads `.tfcoil.dcond[6]` as the superconductor density.
    """

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
    """`i_tf_sc_mat == DURHAM_REBCO` (8) -- Durham REBCO.

    Reads `.tfcoil.dcond[7]` as the superconductor density.
    """

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
