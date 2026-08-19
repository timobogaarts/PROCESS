"""Pure-functional port of `process/models/stellarator/coils/mass.py` (registry #12).

Audit record: `functional_process/models/stellarator/coils/mass.md`. The source's
`calculate_coils_mass` orchestrates 8 sub-functions (`casing`, `ground_insulation`,
`superconductor`, `copper`, `conduit_steel`, `conduit_insulation`, `total_conductor`,
`total_coil`), each writing one `data.tfcoil.*` field that a later sub-function reads
straight back off `data` -- unconditional, unbranched, same-call produce-then-consume, so
this is `local-intermediate` exactly like `stellarator_D_structure.md`'s `aintmass`
chain, just one file over. Ported here as one straight-line function with ordinary
Python locals instead of eight `data`-mediated steps.

`superconductor()`'s `data.tfcoil.dcond[data.tfcoil.i_tf_sc_mat - 1]`
(`process/models/stellarator/coils/mass.py:88`) is a data-table lookup (material
density), not a formula branch. The *pure function* below still takes it as one
already-indexed scalar argument, `den_tf_sc_material` -- that part is unchanged.

**What changed** (MDA triage, `_audit/next_steps.md` §8.1, row
`.tfcoil.den_tf_sc_material`): **`CoilsMass`'s `Input` no longer mints
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
not ports" it is resolved when the graph is assembled, not carried as a port -- and an
`Input` default is fixed at class-definition time, so the index it selects is fixed with
it. `CoilsMass` below is therefore the `i_tf_sc_mat == 1` (ITER Nb3Sn) arm; see
`I_TF_SC_MAT_ITER_NB3SN`.
"""

from cottax.interfaces.pytree_namespace_module import ExplicitFunction, Input, Output

from process.core import constants

I_TF_SC_MAT_ITER_NB3SN = 1
"""`i_tf_sc_mat`'s ITER-Nb3Sn value, and PROCESS's own default
(`process/data_structure/tfcoil_variables.py:246`). `tests/regression/input_files/
stellarator_helias.IN.DAT:235` sets it explicitly to the same 1, so the MDA harness's run
selects `dcond[0] == 6080.0` (`tfcoil_variables.py:158`).

The same value is already hardcoded one node over, at graph assembly, for the same switch
(`WindingPackIntersectInputs(i_tf_sc_mat=1)` in `functional_process/total_process.py`).
A different material needs a sibling node class overriding only `den_tf_sc_material`'s
`Input`, in the style `coils.py` already uses to give `jcrit_from_material` one node
class per `i_tf_sc_mat` value (`coils.py:123,178,232,267,309,347,384,421`).
"""


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
    """cottax node: `calculate_coils_mass`, the `i_tf_sc_mat == 1` arm.

    Every `Input` is a real `DataStructure` field. `den_tf_sc_material` reads
    `.tfcoil.dcond[0]` -- see the module docstring for why the index is baked in here
    rather than passed, and `I_TF_SC_MAT_ITER_NB3SN` for which value it is.

    Its two winding-pack area `Input`s (`.tfcoil.a_tf_wp_with_insulation`/
    `.a_tf_wp_no_insulation`) *are* minted, but by this port's own
    `coils/calculate.py:1136-1137` (`WindingPackTotalSizePost`), which is their producer;
    they are locals in PROCESS (`process/models/stellarator/coils/calculate.py:496-501`,
    the source's own comment: "not global"). They show up as "ungrounded inputs" in the
    MDA harness only because that harness excludes their producer's whole SCC -- see
    `_audit/next_steps.md` §8.1.
    """

    m_tf_coil_case = Output(lambda s: s.tfcoil.m_tf_coil_case)
    m_tf_coil_wp_insulation = Output(lambda s: s.tfcoil.m_tf_coil_wp_insulation)
    m_tf_coil_superconductor = Output(lambda s: s.tfcoil.m_tf_coil_superconductor)
    m_tf_coil_copper = Output(lambda s: s.tfcoil.m_tf_coil_copper)
    m_tf_wp_steel_conduit = Output(lambda s: s.tfcoil.m_tf_wp_steel_conduit)
    m_tf_coil_wp_turn_insulation = Output(
        lambda s: s.tfcoil.m_tf_coil_wp_turn_insulation
    )
    m_tf_coil_conductor = Output(lambda s: s.tfcoil.m_tf_coil_conductor)
    m_tf_coils_total = Output(lambda s: s.tfcoil.m_tf_coils_total)

    def __call__(
        self,
        a_tf_wp_with_insulation=Input(lambda s: s.tfcoil.a_tf_wp_with_insulation),
        a_tf_wp_no_insulation=Input(lambda s: s.tfcoil.a_tf_wp_no_insulation),
        len_tf_coil=Input(lambda s: s.tfcoil.len_tf_coil),
        a_tf_coil_inboard_case=Input(lambda s: s.tfcoil.a_tf_coil_inboard_case),
        den_tf_coil_case=Input(lambda s: s.tfcoil.den_tf_coil_case),
        den_tf_wp_turn_insulation=Input(lambda s: s.tfcoil.den_tf_wp_turn_insulation),
        n_tf_coil_turns=Input(lambda s: s.tfcoil.n_tf_coil_turns),
        a_tf_turn_cable_space_no_void=Input(
            lambda s: s.tfcoil.a_tf_turn_cable_space_no_void
        ),
        f_a_tf_turn_cable_space_extra_void=Input(
            lambda s: s.tfcoil.f_a_tf_turn_cable_space_extra_void
        ),
        f_a_tf_turn_cable_copper=Input(lambda s: s.tfcoil.f_a_tf_turn_cable_copper),
        a_tf_wp_coolant_channels=Input(lambda s: s.tfcoil.a_tf_wp_coolant_channels),
        den_tf_sc_material=Input(lambda s: s.tfcoil.dcond[I_TF_SC_MAT_ITER_NB3SN - 1]),
        a_tf_turn_steel=Input(lambda s: s.tfcoil.a_tf_turn_steel),
        den_steel=Input(lambda s: s.fwbs.den_steel),
        a_tf_coil_wp_turn_insulation=Input(
            lambda s: s.tfcoil.a_tf_coil_wp_turn_insulation
        ),
        n_tf_coils=Input(lambda s: s.tfcoil.n_tf_coils),
    ):
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
