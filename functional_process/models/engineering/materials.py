"""Pure-functional port of `process/models/engineering/materials.py` (partial).

One function: `eurofer97_thermal_conductivity`, the cubic fit for the thermal
conductivity of the EUROFER97 reduced-activation steel the first wall is made of.

**Only one, and the other two are deliberately absent.** PROCESS files
`calculate_tresca_stress` and `calculate_von_mises_stress` in the same module; both are
already ported -- `functional_process/cottax/pfcoil/stresses.py:313` and `:334`, with
`models/tfcoil/stress.py:862`'s `tresca_stress` a second call site of the same formula.
They landed there because their callers are stress models and the shared-helper
convention had not yet reached this file. Re-porting them here would give one formula
two homes, which is worse than the asymmetry; whoever consolidates the stress packages
should lift them into this module then, and this docstring is the note asking for it.

`eurofer97_thermal_conductivity` has exactly one caller in `process/`:
`models/fw.py:487`, inside `FirstWall.fw_temp` -- so unlike `pumping.py`'s three it is
*not* here by the two-caller rule. It is here because it mirrors PROCESS's own filing
and because it belongs to the same dormant chain: `fw_temp` is reached only via
`.fwbs.i_p_coolant_pumping == 2` (`MECHANICAL`), which no tracked regression input
selects. See `pumping.py`'s docstring for why porting the chain's arithmetic does not
lift `indat.py`'s refusal of that arm -- CoolProp does.

Audit record: `functional_process/_audit/units/models/engineering/materials.md`.
"""

__all__ = ["eurofer97_thermal_conductivity"]


def eurofer97_thermal_conductivity(*, temp, fw_th_conductivity):
    """Thermal conductivity of EUROFER97 at temperature `temp`. Ports
    `process/models/engineering/materials.py:14-49`, unchanged.

    Parameters
    ----------
    temp :
        Property temperature (K).
    fw_th_conductivity :
        Thermal conductivity of the first wall material at 293 K (W/m/K). The fit is
        normalised by `28.34`, the cubic's own value at 293 K, so this argument scales
        the whole curve to whatever the input file declares.

    Returns
    -------
    :
        Thermal conductivity (W/m/K).

    Notes
    -----
    Valid to about 800 K. PROCESS applies no guard and extrapolates freely -- its own
    unit test evaluates the fit at 1900 K -- so this port does not guard either.

    A plain polynomial: no division by a computed quantity, no fractional power, no
    branch. Nothing for `safe_math` to cover, and no `jnp` import needed -- the
    expression traces on whatever array type the caller passes in.
    """
    return (
        (5.4308 + 0.13565 * temp - 0.00023862 * temp**2 + 1.3393e-7 * temp**3)
        * fw_th_conductivity
        / 28.34
    )
