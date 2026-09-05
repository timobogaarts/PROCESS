"""Pure-functional port of the tokamak TF quench chain --
`CICCSuperconductingTFCoil.quench_heat_protection_current_density`
(`process/models/tfcoil/superconducting.py:1298-1379`) and the material physics it calls
into (`process/models/tfcoil/quench.py`).

Audit record: `functional_process/_audit/units/models/tfcoil/quench.md`.

## The headline: the dump voltage never reaches CoolProp

`_audit/tokamak_boundary.md` § `.tokamak.cicc_superconducting_tf_coil` schedules this
whole slot behind the unresolved CoolProp wrapping policy on the grounds that
*"`.tfcoil.v_tf_coil_dump_quench_kv` above is on that chain"*. **It is not**, and the
evidence is one function body. `quench_heat_protection_current_density` returns a pair
and the two halves are computed independently:

```python
# process/models/tfcoil/superconducting.py:1357-1360
v_tf_dump_voltage_peak = 2.0e0 * e_tf_coil_magnetic_stored / (t_tf_quench_dump * c_tf_turn)

# process/models/tfcoil/superconducting.py:1362-1377
j_tf_wp_quench_protection_max = (a_tf_turn_cable_space / a_tf_turn
    * quench.calculate_quench_protection_current_density(...))   # <- the only CoolProp path
```

`.tfcoil.v_tf_coil_dump_quench_kv` is `v_tf_dump_voltage_peak / 1.0e3`
(`superconducting.py:2793-2795`) and depends on exactly three variables, none of them a
material property. It is fully ported below, with no seam and no approximation.

`.tfcoil.j_tf_wp_quench_heat_max` is the one that reaches helium, and it is **not** on
this slot's boundary list at all -- nothing in the currently-assembled tokamak graph
reads it.

## The seam, for whoever resolves the CoolProp policy

Everything up to the boundary is ported: the five copper-resistivity pieces, the copper
and Nb3Sn heat capacities, the integrand, the Gauss-Legendre quadrature and the
hotspot-criterion formula itself. **The helium properties enter as ordinary arguments**
-- `den_helium` and `cp_helium`, evaluated at the quadrature nodes -- so the port is a
pure, traceable function of everything except two numbers per node, and those two
numbers are the entire CoolProp surface.

Measured on the reference configuration (`tests/regression/input_files/
large_tokamak_eval.IN.DAT` plus `tfcoil_variables.py` defaults; probe in the audit
record), one call to `calculate_quench_protection_current_density` makes exactly:

| | |
|---|---|
| fluid | `He`, only |
| properties | `C` (isobaric specific heat, J/kg/K) and `D` (density, kg/m3) -- nothing else |
| state variables | `("T", T, "P", P)`, always |
| pressure | `600000.0` Pa exactly, a hardcoded literal (`process/models/tfcoil/quench.py:301,382,448`), never an input |
| temperatures | the 75 Gauss-Legendre nodes of `[tftmp, temp_tf_conductor_quench_max]` = `[4.75, 150.0]` K, i.e. `4.7868...` to `149.9631...` K |
| calls | 150 `PropsSI` (75 x 2) on the **first** evaluation, **0** on every later one |

The zero is the important number. The CoolProp wrapper -- PROCESS's
`process/core/coolprop_interface.py`, vendored verbatim as
`functional_process/fluid_properties.py` -- memoises each
property on its input tuple (`@cache`), and the quadrature grid depends only on `tftmp`
and `temp_tf_conductor_quench_max` -- **neither is written by any model and neither is an
iteration variable** (`grep` over `process/models/**` finds no assignment to either;
`process/core/solver/iteration_variables.py` lists neither; only `t_tf_superconductor_quench`,
ID 56, is an unknown, and it enters through `1/(0.5*tau + t_detect)` *outside* the
integrals). So on this run the helium property surface is **a frozen table of 150
constants**, not a function the optimiser moves along.

That makes the resolution cheap and removes the need to choose between the three
options the dispatch brief listed: no fit and no interpolation table is needed for
`large_tokamak_eval`, because a 75-node lookup evaluated once is already exact. A
JAX-traceable fit or an interpolation in `T` only becomes necessary if a future
configuration makes `tftmp` or `temp_tf_conductor_quench_max` an unknown, and a
`P`-dependence is never needed while the pressure stays a literal.

**This module deliberately declares no node for the current density.** `den_helium`/
`cp_helium` have no `VarPath` -- they are not `DataStructure` fields -- so a node would
have to mint two, which is a policy call for the orchestrator, not for this port
(`traceability_policy.md` § "Non-traceable external calls" defers exactly this). The
pure functions are here, tested against PROCESS, ready for whichever binding is chosen.

## Two more things worth knowing before wrapping it

- **`process/models/tfcoil/quench.py:18` sets `COPPER_DENSITY = 8960.0`, while
  `process/core/constants.py:289` sets `DEN_COPPER = 8900.0`**, and the mass chain in
  `functional_process/cottax/tfcoil/superconducting.py` uses the latter. Two different
  copper densities in one coil model. Ported faithfully (each formula keeps its own),
  recorded as defect **D3**.
- `calculate_quench_protection_current_density` **clips the fluence** to `[0, 1.5e23]`
  with a warning (`quench.py:533-537`), because *"default fluence is too high for this
  model"*. The reference run's `flu_tf_neutron_fast_max = 1e22`
  (`large_tokamak_eval.IN.DAT:387`) is inside the range, so the clip is inert there --
  but it is a real kink in the derivative and is ported as `jnp.clip`, not dropped.
"""

import equinox as eqx
from cottax.interfaces.pytree_namespace_module import ExplicitFunction, From, OutputInto

from functional_process.paths import constraints, superconducting_tfcoil, tfcoil
from functional_process.models.tfcoil.quench import (
    QUENCH_HELIUM_PRESSURE_PA,  # noqa: F401 -- re-exported for tests
    calculate_quench_protection_current_density,  # noqa: F401 -- re-exported for tests
    calculate_tf_coil_quench_heat_current_density,
    copper_electrical_resistivity,  # noqa: F401 -- re-exported for tests
    copper_irradiation_resistivity,  # noqa: F401 -- re-exported for tests
    copper_magneto_resistivity,  # noqa: F401 -- re-exported for tests
    copper_rrr_resistivity,  # noqa: F401 -- re-exported for tests
    copper_specific_heat_capacity,  # noqa: F401 -- re-exported for tests
    helium_properties_at_quench_nodes,  # noqa: F401 -- re-exported for indat.py / tests
    j_tf_wp_quench_heat_max,  # noqa: F401 -- re-exported for tests
    nb3sn_specific_heat_capacity,  # noqa: F401 -- re-exported for tests
    quench_integrals,  # noqa: F401 -- re-exported for tests
    quench_quadrature_temperatures,  # noqa: F401 -- re-exported for tests
    tf_dump_voltage_peak,  # noqa: F401 -- re-exported for tests
    v_tf_coil_dump_quench_kv,
)


class TfCoilQuenchHeatCurrentDensity(ExplicitFunction):
    """cottax node: `.tfcoil.j_tf_wp_quench_heat_max`, constraint 35's read.

    ## The CoolProp policy call, made (2026-08-27)

    `quench.md` OQ1 listed three resolutions for the helium property boundary and
    declined to pick one. This class picks **(a) -- the property table is a constant of
    the run, resolved at graph-assembly time and carried as a static field** -- and the
    reason is the measurement that record already contains rather than a preference:

    - The quadrature grid is a function of `.tfcoil.tftmp` and
      `.tfcoil.temp_tf_conductor_quench_max` and nothing else.
    - **Neither is written by any model** (`grep` over `process/models/**` finds no
      assignment to either; the only site is the read at `superconducting.py:2785`) and
      **neither is an iteration variable**. So the states CoolProp is asked about do not
      move during a solve, a scan point, or a gradient sweep.
    - Therefore the 150 numbers are constants of the machine in the same sense that
      `models/stellarator/preset_config.py`'s `machine_config` is -- and they are
      carried the same way, as an `eqx.field(static=True)` filled by `indat.py` from
      the file's own values.

    What that buys, against the two options not taken: a `pure_callback` would put a
    host round-trip inside a `jit` region to fetch numbers that cannot change, and a fit
    or interpolation would introduce an approximation where an exact table is already
    available at zero marginal cost. Nothing is approximated here, and no CoolProp call
    happens inside any traced or differentiated region.

    ## What the decision costs, stated rather than hidden

    `tftmp` and `temp_tf_conductor_quench_max` are **not reads of this node**, so the
    graph carries no edge from them and `jax` sees no derivative with respect to either.
    That is exactly right while they are inputs and exactly wrong the moment one is not,
    which is why `indat.py` **refuses to assemble** a machine whose `ixc` contains either
    (`_quench_helium_table`). The refusal is the thing that makes the static field safe;
    without it this would be the same shape of defect as the `dcond[0]` bake
    `superconducting.md` records for the mass slot.

    The harness contracts already excluded both from differentiation for this same
    reason (`tests/functional_process/models/tfcoil/test_quench.py`'s `_SEAM_STATICS`),
    so the node's declared reads and the case's differentiated arguments now say the
    same thing.
    """

    tftmp: float = eqx.field(static=True)
    """The helium peak-field temperature, as a static value rather than a read.

    **It is not only the property grid that depends on it** -- the quadrature interval
    `[tftmp, temp_tf_conductor_quench_max]` sets the nodes and the weights too, so
    freezing the table without freezing the interval would evaluate the integrand at
    temperatures the properties do not belong to. Both endpoints are static together, or
    neither.

    Named for the `DataStructure` field rather than for the pure function's parameter
    (`temp_he_peak`) **on purpose**: `mda_harness.switch_audit` resolves a static
    kwarg's backing field by name, so spelling it `tftmp` makes the frozen value
    value-checked against the converged run automatically, on every harness run. A
    static field with no resolvable name is a value nothing can check."""

    temp_tf_conductor_quench_max: float = eqx.field(static=True)
    """The hotspot temperature limit. See `tftmp`, including on the naming."""

    den_helium_at_nodes: tuple = eqx.field(static=True)
    """Helium density at `quench_quadrature_temperatures(...)`, from
    `helium_properties_at_quench_nodes`. A `tuple` and not an array because a static
    field has to be hashable -- it is a jit cache key."""

    cp_helium_at_nodes: tuple = eqx.field(static=True)
    """Helium isobaric specific heat at the same nodes."""

    j_tf_wp_quench_heat_max = OutputInto(tfcoil)

    def __call__(
        self,
        a_tf_turn_cable_space_no_void=From(tfcoil),
        a_tf_turn=From(tfcoil),
        t_tf_superconductor_quench=From(tfcoil),
        b_tf_inboard_peak_with_ripple=From(tfcoil),
        f_a_tf_turn_cable_copper=From(tfcoil),
        f_a_tf_turn_cable_space_cooling=From(superconducting_tfcoil),
        rrr_tf_cu=From(tfcoil),
        t_tf_quench_detection=From(tfcoil),
        flu_tf_neutron_fast_max=From(constraints),
    ):
        return calculate_tf_coil_quench_heat_current_density(
            a_tf_turn_cable_space_no_void,
            a_tf_turn,
            t_tf_superconductor_quench,
            b_tf_inboard_peak_with_ripple,
            f_a_tf_turn_cable_copper,
            f_a_tf_turn_cable_space_cooling,
            self.tftmp,
            self.temp_tf_conductor_quench_max,
            rrr_tf_cu,
            t_tf_quench_detection,
            flu_tf_neutron_fast_max,
            self.den_helium_at_nodes,
            self.cp_helium_at_nodes,
        )


class TfCoilDumpQuenchVoltage(ExplicitFunction):
    """cottax node: `.tfcoil.v_tf_coil_dump_quench_kv`, one of the slot's ten reads.

    Three reads, no switch, no property boundary. The companion output of the same
    PROCESS function, `.tfcoil.j_tf_wp_quench_heat_max`, gets **no node here**: it needs
    helium density and specific heat, which have no `VarPath`, and minting two is the
    orchestrator's call. See the module docstring.
    """

    v_tf_coil_dump_quench_kv = OutputInto(tfcoil)

    def __call__(
        self,
        e_tf_coil_magnetic_stored=From(tfcoil),
        t_tf_superconductor_quench=From(tfcoil),
        c_tf_turn=From(tfcoil),
    ):
        return v_tf_coil_dump_quench_kv(
            e_tf_coil_magnetic_stored=e_tf_coil_magnetic_stored,
            t_tf_superconductor_quench=t_tf_superconductor_quench,
            c_tf_turn=c_tf_turn,
        )
