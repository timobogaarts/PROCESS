"""PROCESS's constraint equations as model nodes.

One `constraint_<id>` in `functional_process.models.constraints` is a ported function
like any other; this module is the thin cottax layer over it -- the reads resolved
against the graph the run actually holds, the switch arguments frozen at assembly, and
the node owning `.constraints.c<id>`, element 1 of the function's
`(residual, normalised_residual, value, bound)` tuple.

**A constraint's value is a model's output**, so it lives in the port's own namespace
and not in one cottax opened: `.constraints.c<id>`, beside the limits PROCESS already
keeps in that area. `^cond` was a mint over a place nothing else claimed, which said
the value was fabricated when it is computed.

**A constraint is declared where it is computed.** The node is a cottax
`ConstraintFunction`, the declaration form for a body whose outputs are *held* against
zero: `holds = Eq` for an equality, `Le` for an inequality, and the requirement that
says so is the declaration's own, bound beside it at `^require.Constraint<id>`. Nothing
states it a second time; an architecture absorbs it into the optimiser its conditions
reach, or `Determine` closes it over a variable.
"""

import functools
import inspect

from cottax.interfaces import Le, Minted
from cottax.interfaces.pytree_namespace_module import (
    Area,
    ConstraintFunction,
    FromExactly,
    Output,
)
from cottax.pytree.path import DictKey, GetAttrKey, NodePath, SequenceKey, VarPath

from functional_process.models import constraints as ported_constraints
from functional_process.vocabulary.input_variables import INPUT_VARIABLES

REQUIREMENT = Minted("require")
"""How the requirement beside a constraint is named: `.Constraint5` ->
`^require.Constraint5`. `ConstraintFunction`'s own default, written down here because
`requirement_place` spells the same name without building the declaration."""


REFERENCE_SWITCH_VALUES = {
    "i_rad_loss": 1,  # `.physics.i_rad_loss`
    # `.physics.i_plasma_ignited`, `stellarator_helias.IN.DAT:126`
    "i_plasma_ignited": 1,
    "i_beta_component": 0,  # `.physics.i_beta_component`
    "istell": 6,  # `.stellarator.istell`, `stellarator_helias.IN.DAT:137`
}
"""Static switch arguments of the reference run's active constraints, and their values.
"""


SWITCH_PARAMETER_NAMES = (
    "bkt_life_csf",
    "i_beta_component",
    "i_cp_lifetime",
    "i_density_limit",
    "i_plant_availability",
    "i_plasma_ignited",
    "i_q95_fixed",
    "i_rad_loss",
    "i_tf_bucking",
    "i_tf_inside_cs",
    "i_tf_sup",
    "ibkt_life",
    "ireactor",
    "istell",
    "itart",
)
"""Every parameter name that is a **switch** anywhere in the ported constraint/objective
surface -- the union of the `static_argnames` the `Tier1Contract`s in
`functional_process/tests/cottax/core/solver/test_constraints.py` and
`test_objectives.py` declare.
"""


NON_INPUT_FIELDS = {
    "available_radial_space": "build",
    "b_cs_peak_flat_top_end": "pf_coil",
    "b_cs_peak_pulse_start": "pf_coil",
    "b_plasma_total": "physics",
    "b_tf_inboard_peak_with_ripple": "tfcoil",
    "beta_beam": "physics",
    "beta_fast_alpha": "physics",
    "beta_poloidal_eps": "physics",
    "beta_poloidal_vol_avg": "physics",
    "beta_thermal_vol_avg": "physics",
    "beta_toroidal_vol_avg": "physics",
    "big_q_plasma": "current_drive",
    "bktcycles": "costs",
    "c_tf_total": "tfcoil",
    "cdirt": "costs",
    "coe": "costs",
    "concost": "costs",
    "coppera_m2": "rebco",
    "cplife": "costs",
    "dr_fw_outboard": "build",
    "dx_tf_inboard_out_toroidal": "tfcoil",
    "eps": "physics",
    "eta_cd_norm_hcd_primary": "current_drive",
    "f_p_beam_shine_through": "current_drive",
    "f_p_plasma_separatrix_rad": "physics",
    "f_pden_alpha_electron_mw": "physics",
    "f_pden_alpha_ions_mw": "physics",
    "f_t_alpha_energy_confinement": "physics",
    "flu_tf_neutron_fast_peak": "fwbs",
    "fzmin": "reinke",
    "j_cs_critical_flat_top_end": "pf_coil",
    "j_cs_critical_pulse_start": "pf_coil",
    "j_cs_pulse_start": "pf_coil",
    "j_tf_wp": "tfcoil",
    "j_tf_wp_critical": "tfcoil",
    "j_tf_wp_quench_heat_max": "tfcoil",
    "life_blkt_fpy": "fwbs",
    "life_div_fpy": "costs",
    "n_beam_decay_lengths_core": "current_drive",
    "n_charge_plasma_effective_vol_avg": "physics",
    "n_cycle": "cs_fatigue",
    "n_iter_vacuum_pumps": "vacuum",
    "nd_beam_ions": "physics",
    "nd_beam_ions_out": "physics",
    "nd_plasma_electron_line": "physics",
    "nd_plasma_electron_on_axis": "physics",
    "nd_plasma_electrons_max": "physics",
    "nd_plasma_ions_total_vol_avg": "physics",
    "p_cp_resistive_mw": "tfcoil",
    "p_cryo_plant_electric_mw": "heat_transport",
    "p_div_bt_q_aspect_rmajor_mw": "physics",
    "p_fusion_total_mw": "physics",
    "p_hcd_injected_electrons_mw": "current_drive",
    "p_hcd_injected_ions_mw": "current_drive",
    "p_hcd_injected_total_mw": "current_drive",
    "p_l_h_threshold_mw": "physics",
    "p_plant_electric_net_mw": "heat_transport",
    "p_plasma_heating_total_mw": "physics",
    "p_plasma_separatrix_mw": "physics",
    "p_plasma_separatrix_rmajor_mw": "physics",
    "p_tf_leg_resistive_mw": "tfcoil",
    "pden_alpha_total_mw": "physics",
    "pden_electron_transport_loss_mw": "physics",
    "pden_ion_electron_equilibration_mw": "physics",
    "pden_ion_transport_loss_mw": "physics",
    "pden_non_alpha_charged_mw": "physics",
    "pden_plasma_core_rad_mw": "physics",
    "pden_plasma_ohmic_mw": "physics",
    "pden_plasma_rad_mw": "physics",
    "peakpoloidalpower": "pf_power",
    "pflux_fw_neutron_mw": "physics",
    "pflux_fw_rad_max_mw": "constraints",
    "plasma_current": "physics",
    "powerht_constraint": "stellarator",
    "powerscaling_constraint": "stellarator",
    "psolradmw": "physics",
    "ptfnucpm3": "fwbs",
    "q95_min": "physics",
    "radius_beam_tangency": "current_drive",
    "radius_beam_tangency_max": "current_drive",
    "rbld": "build",
    "required_radial_space": "build",
    "rminor": "physics",
    "sig_tf_case": "tfcoil",
    "sig_tf_cs_bucked": "tfcoil",
    "sig_tf_wp": "tfcoil",
    "srcktpm": "pf_power",
    "str_wp": "tfcoil",
    "stress_shear_cs_peak": "pf_coil",
    "t_current_ramp_up_min": "constraints",
    "t_plant_pulse_total": "times",
    "tcpav2": "tfcoil",
    "temp_cp_peak": "tfcoil",
    "temp_croco_quench": "tfcoil",
    "temp_cs_superconductor_margin": "pf_coil",
    "temp_fw_peak": "fwbs",
    "temp_plasma_electron_density_weighted_kev": "physics",
    "temp_plasma_ion_density_weighted_kev": "physics",
    "temp_tf_superconductor_margin": "tfcoil",
    "tfcmw": "tfcoil",
    "toroidalgap": "tfcoil",
    "v_tf_coil_dump_quench_kv": "tfcoil",
    "vol_plasma": "physics",
    "vs_cs_pf_total_pulse": "pf_coil",
    "vs_cs_pf_total_ramp": "pf_coil",
    "vs_plasma_ramp_required": "physics",
    "vs_plasma_total_required": "physics",
    "vv_stress_quench": "superconducting_tfcoil",
}
"""`field name -> DataStructure area` for every name the ported constraint/objective
layer can ask for that is **not** a declared PROCESS input.
"""


class Resolver:
    """`parameter name -> VarPath`, graph first, PROCESS's declared inputs second.

    A constraint names its reads by parameter name, the way every ported model does, so
    the place each one means is looked up: a variable this graph already holds when its
    **leaf name is unique** there, else the `DataStructure` area the name is declared
    in. Ambiguity is refused rather than guessed at.
    """

    def __init__(self, graph_variables):
        self.by_name = {}
        for var in graph_variables:
            keys = var.segments
            if len(keys) == 2 and all(isinstance(k, GetAttrKey) for k in keys):
                self.by_name.setdefault(keys[-1].name, set()).add(var)

    def __call__(self, name: str) -> VarPath:
        """Where `name` lives.

        Raises
        ------
        ValueError
            If the graph holds several variables with that leaf name, or none does and
            the name is no declared input either.
        """
        hits = self.by_name.get(name)
        if hits and len(hits) == 1:
            return next(iter(hits))
        if hits:
            raise ValueError(
                f"{name!r} names {len(hits)} variables in the graph "
                f"({sorted(v.spelling for v in hits)}) -- resolution is by unique "
                f"name, so this one has to be given explicitly"
            )
        area = NON_INPUT_FIELDS.get(name)
        if area is None:
            declaration = INPUT_VARIABLES.get(name)
            # `ixc`/`icc` carry `module=None`: they are the problem statement, not a
            # field, so they are no more resolvable than an undeclared name.
            area = None if declaration is None else declaration.module
        if area is not None:
            return VarPath((GetAttrKey(area), GetAttrKey(name)))
        raise ValueError(
            f"{name!r} resolves to nothing, and the two halves of that are different "
            f"failures. (1) No node in this graph owns or reads it -- it is not part of "
            f"*this* machine's dataflow, and the same name may well be produced on "
            f"another device configuration, so check the machine before the spelling. "
            f"(2) It is not a declared PROCESS input either "
            f"(`vocabulary/input_variables.py`, PROCESS's own `INPUT_VARIABLES`), so it "
            f"cannot be a constraint bound read off the boundary -- which leaves a "
            f"typo, or a quantity an unported model computes, in which case it "
            f"belongs in "
            f"`NON_INPUT_FIELDS` with its `DataStructure` area."
        )


def bind(fn, resolve, switches):
    """`(switch pairs, read names, reads)` for one ported constraint/objective function.

    A parameter named in `switches` is frozen at assembly -- the port's rule that a
    model never reads an integer switch -- and every other one is a read.
    """
    parameters = list(inspect.signature(fn).parameters)
    static = tuple((p, switches[p]) for p in parameters if p in switches)
    read = [p for p in parameters if p not in switches]
    return static, read, tuple(resolve(p) for p in read)


def condition_place(cid: int) -> VarPath:
    """Where constraint `cid`'s value is computed: `.constraints.c<id>`."""
    return VarPath((GetAttrKey("constraints"), GetAttrKey(f"c{cid}")))


def constraint_place(cid: int) -> NodePath:
    """Where constraint `cid`'s body binds: `Constraint<id>`."""
    return NodePath((GetAttrKey(f"Constraint{cid}"),))


def requirement_place(cid: int) -> NodePath:
    """Where constraint `cid`'s requirement binds: `^require.Constraint<id>` -- the
    declaration's own name for it, spelled without building the declaration.
    """
    return REQUIREMENT(constraint_place(cid))


def _where(place):
    """`place` as something `FromExactly` and `Output` take: the area it sits in, then
    the one key that reaches it.

    A declaration names a place by the access path that reaches it, and these are
    places resolved against a graph -- so the path is walked back into the recording
    the two calls expect.

    Raises
    ------
    ValueError
        If `place` is the root, or ends in a key kind no declaration can spell.
    """
    keys = place.segments
    if not keys:
        raise ValueError(f"{place!r} names the root itself, not a place inside it")
    area, last = Area(keys[:-1]), keys[-1]
    if isinstance(last, GetAttrKey):
        return getattr(area, last.name)
    if isinstance(last, SequenceKey):
        return area[last.idx]
    if isinstance(last, DictKey):
        return area[last.key]
    raise ValueError(f"{place!r} ends in {last!r}, which no declaration can spell")


@functools.cache
def _constraint_class(cid: int, holds, read_names: tuple, reads: tuple, switches: tuple):
    """The `ConstraintFunction` class for one constraint, its reads and its switches.

    Cached on exactly what it is made of, because the class **is** the body: a
    declaration compares by its type and its fields, and a graph is a jit cache key, so
    assembling the same problem twice must give the same class back.

    The `__call__` is synthesised the way `cottax.wraps.WrapsFunction` synthesises one
    -- a signature whose parameter defaults are the reads -- since a constraint's reads
    are resolved against the graph and cannot be written in a class body.
    """
    fn = getattr(ported_constraints, f"constraint_{cid}")
    static = dict(switches)

    def call(self, *args, **kwargs):
        bound = inspect.signature(type(self).__call__).bind(self, *args, **kwargs)
        bound.apply_defaults()
        arguments = {k: v for k, v in bound.arguments.items() if k != "self"}
        arguments.update(static)
        return fn(**arguments)[1]

    call.__signature__ = inspect.Signature([
        inspect.Parameter("self", inspect.Parameter.POSITIONAL_OR_KEYWORD),
        *(
            inspect.Parameter(
                name,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                default=FromExactly(_where(place)),
            )
            for name, place in zip(read_names, reads, strict=True)
        ),
    ])
    call.__name__ = "__call__"
    call.__doc__ = (
        f"`{fn.__module__}.{fn.__name__}`'s normalised residual -- element 1 of "
        f"`(residual, normalised_residual, value, bound)`, ports declared above."
    )
    return type(
        f"Constraint{cid}",
        (ConstraintFunction,),
        {
            "__module__": __name__,
            "__doc__": f"PROCESS constraint {cid}, held by `{holds.__name__}`.",
            "holds": holds,
            "__call__": call,
            condition_place(cid).leaf.name: Output(_where(condition_place(cid))),
        },
    )


def constraint_declaration(graph_variables, cid: int, holds=Le, switches=None):
    """The declaration of PROCESS constraint `cid` over a graph's variables.

    A `ConstraintFunction`: the body owns `.constraints.c<id>` at
    `Constraint<id>`, and the requirement holding it against zero by `holds` is bound
    beside it at `^require.Constraint<id>`.

    Raises
    ------
    ValueError
        If no `constraint_<id>` is ported, or one of its parameters names no place.
    """
    switches = REFERENCE_SWITCH_VALUES if switches is None else switches
    fn = getattr(ported_constraints, f"constraint_{cid}", None)
    if fn is None:
        raise ValueError(
            f"constraint {cid} is active in this run but `models/constraints.py` has "
            f"no `constraint_{cid}`"
        )
    try:
        static, read, reads = bind(fn, Resolver(graph_variables), switches)
    except ValueError as e:
        raise ValueError(
            f"constraint {cid} cannot be assembled: {e}. An optimiser missing one of "
            f"PROCESS's active constraints solves a different problem -- leave it out "
            f"on purpose and have it reported, or give the name a place"
        ) from e
    return _constraint_class(cid, holds, tuple(read), reads, static)(
        mint_requirement=REQUIREMENT
    )
