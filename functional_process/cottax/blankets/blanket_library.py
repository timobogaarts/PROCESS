"""Pure-functional port of `process/models/blankets/blanket_library.py`'s tokamak
`component_volumes` chain.

**Why this file exists at all.** `blankets/blanket_library.py` is one of the three files
`tokamak_call_surface.md` §A found *reached with no `.run()` in `caller.py`*:
`models.blanket_library` is constructed at `main.py:678` and never called, and the file
runs only because `CCFE_HCPB(OutboardBlanket, InboardBlanket)` (`hcpb.py:25`) inherits
from `BlanketLibrary` (`blanket_library.py:56`). Fourteen of its functions are entered on
the reference tokamak run; this port covers the **four** of them that lie on the minimal
closure producing `.tokamak.ccfe_hcpb`'s boundary variables -- everything downstream of
`.fwbs.vol_blkt_total`, which `CCFE_HCPB.component_masses` needs and nothing else in the
tokamak surface produces.

The other ten entered functions (`set_blanket_module_geometry`,
`pipe_hydraulic_diameter`, the four poloidal-segment/module-geometry helpers, the two
poloidal-plasma-angle helpers) were **deliberately out of scope**: measured, every one of
their writes lands in `.blanket.*` or in `.fwbs.b_bz_liq`/`a_bz_liq`/
`radius_blkt_channel*`, and none of those reaches any of the sixteen variables
`_audit/tokamak_boundary.md` §`.tokamak.ccfe_hcpb` lists. See
`_audit/units/models/blankets/blanket_library.md` (read it first) for the evidence table.

**One of the ten is in scope since 2026-08-30**, and the sentence above is exactly why
it was missed: `calculate_blkt_inboard_poloidal_plasma_angle` writes `.blanket.*`, so
the reads-nothing-of-*this*-slot's-boundary test passed -- while
`.tokamak.divertor.heat_flux_split`, a different slot, was reading the field and getting
the cold `0.0`. The test that catches this is asked of the assembled machine
(`boundary.unproduced_but_computed`), not of a slot. Nine remain out of scope, on the
same evidence, and nothing about that evidence rules out the same surprise: what changed
is that there is now a check that would say so.

**Switches.** `component_volumes` (`blanket_library.py:91-94`) chooses D-shaped vs
elliptical blanket geometry on `itart == 1 or i_fw_blkt_vv_shape == D_SHAPED`; the
reference run has `itart = 0` and `i_fw_blkt_vv_shape = 2` (`ELLIPTICAL_SHAPED`), so only
the elliptical arm is ported. `calculate_blkt_half_height` and `apply_coverage_factors`
branch on `n_divertors == 2`; the reference run has `n_divertors = 1`. Every unported arm
is named as UNPORTED in the audit record rather than folded into a `jnp.where` -- the
union-of-arms reads is the invented-edge defect this port exists to remove
(`next_steps.md` §14.2).

2026-08-27 (the double-null wave, for the two spherical-tokamak input files that set
`i_single_null = 0`): **both** `n_divertors` slots are now total -- the double-null arms
of the half-height and of the coverage factors are written beside their single-null
siblings, each a separate occupant with its own reads-set. The half-height's two arms
differ by five reads and the coverage factors' by a literal, and per `next_steps.md`
§14.2 (the `istore` precedent) a literal is enough: a switch value selects an occupant.
`n_divertors` is still a parameter of nothing.

2026-08-27 (the D-shaped wave, same two spherical-tokamak files -- both also set
`i_fw_blkt_vv_shape = 1` and `itart = 1`): the shape decision's D-shaped arm is written
too, so **all four of this file's slots are now total**. `DShapedBlanketAreas` and
`DShapedBlanketVolumes` join the elliptical pair.

**The shape and the divertor count do not interact in this file.** `component_volumes`
(`blanket_library.py:71-165`) runs three consecutive, independent blocks: the half-height
(branches on `n_divertors`), the areas *and* volumes (branch on the shape), and the
coverage factors (branch on `n_divertors` again). Because wave 1 had already split those
three blocks into four separate cottax slots, each slot is keyed on exactly **one**
predicate and no slot needs a shape x divertor-count product. That is a property of the
decomposition, not of PROCESS: `models/fw.py` and `models/vacuum/vacuum.py` keep one
composite node spanning both branches, so those two slots *do* pay the product. See
`fw.py`'s module docstring.

The D-shaped arm reads **no `triang`** where the elliptical arm does, and it reads five
`.build` thicknesses plus `.physics.rminor` where the elliptical arm reads
`r_shld_outboard_outer` and `.physics.rmajor`: the two arms are not the same node under
a parameter, which is why they are occupants.
"""

import jax.numpy as jnp
from cottax.interfaces.pytree_namespace_module import ExplicitFunction, From, OutputInto

from functional_process.models.blankets.blanket_library import (
    apply_coverage_factors_double_null,
    apply_coverage_factors_single_null,
    calculate_blkt_half_height_double_null,
    calculate_blkt_half_height_single_null,
    calculate_blkt_inboard_poloidal_plasma_angle,
    calculate_dshaped_blkt_areas,
    calculate_dshaped_blkt_volumes,
    calculate_elliptical_blkt_areas,
    calculate_elliptical_blkt_volumes,
)
from functional_process.models.engineering.ivc_functions import dshellarea, dshellvol
from functional_process.paths import blanket, build, divertor, fwbs, physics

# ruff's docstring rules treat `__all__` membership as the definition of "public" once
# one is present, so this lists every public name this module resolved before step 2 of
# `_audit/formulas_split.md` moved the pure functions out -- not just `dshellarea`/
# `dshellvol`/`jnp`, which are unused now that their real uses left with the functions
# (see `power/electric_production.py`'s commit for why a partial list is the wrong move).
__all__ = [
    "BlanketAreas",
    "BlanketCoverageFactors",
    "BlanketCoverageFactorsDoubleNull",
    "BlanketCoverageFactorsSingleNull",
    "BlanketHalfHeight",
    "BlanketHalfHeightDoubleNull",
    "BlanketHalfHeightSingleNull",
    "BlanketInboardPoloidalAngle",
    "BlanketVolumes",
    "DShapedBlanketAreas",
    "DShapedBlanketVolumes",
    "EllipticalBlanketAreas",
    "EllipticalBlanketVolumes",
    "ExplicitFunction",
    "From",
    "OutputInto",
    "apply_coverage_factors_double_null",
    "apply_coverage_factors_single_null",
    "blanket",
    "build",
    "calculate_blkt_half_height_double_null",
    "calculate_blkt_half_height_single_null",
    "calculate_blkt_inboard_poloidal_plasma_angle",
    "calculate_dshaped_blkt_areas",
    "calculate_dshaped_blkt_volumes",
    "calculate_elliptical_blkt_areas",
    "calculate_elliptical_blkt_volumes",
    "divertor",
    "dshellarea",
    "dshellvol",
    "fwbs",
    "jnp",
    "physics",
]


class BlanketHalfHeight(ExplicitFunction):
    """The family that owns `.blanket.dz_blkt_half`: one occupant per `n_divertors` arm
    of `BlanketLibrary.calculate_blkt_half_height`.

    Both arms are written (2026-08-27), so this slot is total. They are separate
    occupants and not one node with a `jnp.where` because the double-null arm reads five
    fields fewer -- see `calculate_blkt_half_height_double_null`.
    """


class BlanketHalfHeightSingleNull(BlanketHalfHeight):
    """cottax node: `calculate_blkt_half_height_single_null`. `n_divertors == 1`."""

    dz_blkt_half = OutputInto(blanket)

    def __call__(
        self,
        z_plasma_xpoint_lower=From(build),
        dz_xpoint_divertor=From(build),
        dz_divertor=From(divertor),
        dz_blkt_upper=From(build),
        z_plasma_xpoint_upper=From(build),
        dr_fw_plasma_gap_inboard=From(build),
        dr_fw_plasma_gap_outboard=From(build),
        dr_fw_inboard=From(build),
        dr_fw_outboard=From(build),
    ):
        return calculate_blkt_half_height_single_null(
            z_plasma_xpoint_lower,
            dz_xpoint_divertor,
            dz_divertor,
            dz_blkt_upper,
            z_plasma_xpoint_upper,
            dr_fw_plasma_gap_inboard,
            dr_fw_plasma_gap_outboard,
            dr_fw_inboard,
            dr_fw_outboard,
        )


class BlanketHalfHeightDoubleNull(BlanketHalfHeight):
    """cottax node: `calculate_blkt_half_height_double_null`. `n_divertors == 2`.

    Live on `spherical_tokamak_eval.IN.DAT` and `st_regression.IN.DAT`, both of which
    set `i_single_null = 0` (`:292`, `:638`), from which `init.py:606-617` derives
    `n_divertors = 2`.
    """

    dz_blkt_half = OutputInto(blanket)

    def __call__(
        self,
        z_plasma_xpoint_lower=From(build),
        dz_xpoint_divertor=From(build),
        dz_divertor=From(divertor),
        dz_blkt_upper=From(build),
    ):
        return calculate_blkt_half_height_double_null(
            z_plasma_xpoint_lower,
            dz_xpoint_divertor,
            dz_divertor,
            dz_blkt_upper,
        )


class BlanketAreas(ExplicitFunction):
    """The family that owns the three `.build.a_blkt_*_full_coverage` fields: one
    occupant per arm of `component_volumes`' shape decision
    (`itart == 1 or i_fw_blkt_vv_shape == D_SHAPED`, `blanket_library.py:90-93`).

    Both arms are written (2026-08-27), so this slot is total. They read overlapping but
    unequal sets -- the D-shaped arm reads no `triang` and no outboard build radius --
    which is why a shape *parameter* was never an option.
    """


class BlanketVolumes(ExplicitFunction):
    """The family that owns the three `.fwbs.vol_blkt_*_full_coverage` fields, on the
    same shape predicate as `BlanketAreas` and with the same two arms. Total since
    2026-08-27.

    A separate family from `BlanketAreas` because PROCESS writes the two through two
    separate `@staticmethod`s into two different namespaces, and nothing downstream reads
    an area to get a volume.
    """


class EllipticalBlanketAreas(BlanketAreas):
    """cottax node: `calculate_elliptical_blkt_areas`.

    Occupies the elliptical arm of `component_volumes`' shape decision
    (`itart == 0` and `.fwbs.i_fw_blkt_vv_shape == ELLIPTICAL_SHAPED`).
    """

    a_blkt_inboard_surface_full_coverage = OutputInto(build)
    a_blkt_outboard_surface_full_coverage = OutputInto(build)
    a_blkt_total_surface_full_coverage = OutputInto(build)

    def __call__(
        self,
        rmajor=From(physics),
        rminor=From(physics),
        triang=From(physics),
        r_shld_inboard_inner=From(build),
        dr_shld_inboard=From(build),
        dr_blkt_inboard=From(build),
        r_shld_outboard_outer=From(build),
        dr_shld_outboard=From(build),
        dr_blkt_outboard=From(build),
        dz_blkt_half=From(blanket),
    ):
        return calculate_elliptical_blkt_areas(
            rmajor,
            rminor,
            triang,
            r_shld_inboard_inner,
            dr_shld_inboard,
            dr_blkt_inboard,
            r_shld_outboard_outer,
            dr_shld_outboard,
            dr_blkt_outboard,
            dz_blkt_half,
        )


class DShapedBlanketAreas(BlanketAreas):
    """cottax node: `calculate_dshaped_blkt_areas`.

    Occupies the D-shaped arm (`itart == 1 or i_fw_blkt_vv_shape == D_SHAPED`). Live on
    `spherical_tokamak_eval.IN.DAT` and `st_regression.IN.DAT`, which satisfy the
    disjunction twice over.

    Reads nine fields to the elliptical sibling's ten, and only five are shared --
    see `calculate_dshaped_blkt_areas` for the two-way difference list.
    """

    a_blkt_inboard_surface_full_coverage = OutputInto(build)
    a_blkt_outboard_surface_full_coverage = OutputInto(build)
    a_blkt_total_surface_full_coverage = OutputInto(build)

    def __call__(
        self,
        r_shld_inboard_inner=From(build),
        dr_shld_inboard=From(build),
        dr_blkt_inboard=From(build),
        dr_fw_inboard=From(build),
        dr_fw_plasma_gap_inboard=From(build),
        rminor=From(physics),
        dr_fw_plasma_gap_outboard=From(build),
        dr_fw_outboard=From(build),
        dz_blkt_half=From(blanket),
    ):
        return calculate_dshaped_blkt_areas(
            r_shld_inboard_inner,
            dr_shld_inboard,
            dr_blkt_inboard,
            dr_fw_inboard,
            dr_fw_plasma_gap_inboard,
            rminor,
            dr_fw_plasma_gap_outboard,
            dr_fw_outboard,
            dz_blkt_half,
        )


class EllipticalBlanketVolumes(BlanketVolumes):
    """cottax node: `calculate_elliptical_blkt_volumes`. Same arm as the areas above."""

    vol_blkt_inboard_full_coverage = OutputInto(fwbs)
    vol_blkt_outboard_full_coverage = OutputInto(fwbs)
    vol_blkt_total_full_coverage = OutputInto(fwbs)

    def __call__(
        self,
        rmajor=From(physics),
        rminor=From(physics),
        triang=From(physics),
        r_shld_inboard_inner=From(build),
        dr_shld_inboard=From(build),
        dr_blkt_inboard=From(build),
        r_shld_outboard_outer=From(build),
        dr_shld_outboard=From(build),
        dr_blkt_outboard=From(build),
        dz_blkt_half=From(blanket),
        dz_blkt_upper=From(build),
    ):
        return calculate_elliptical_blkt_volumes(
            rmajor,
            rminor,
            triang,
            r_shld_inboard_inner,
            dr_shld_inboard,
            dr_blkt_inboard,
            r_shld_outboard_outer,
            dr_shld_outboard,
            dr_blkt_outboard,
            dz_blkt_half,
            dz_blkt_upper,
        )


class DShapedBlanketVolumes(BlanketVolumes):
    """cottax node: `calculate_dshaped_blkt_volumes`. Same arm as `DShapedBlanketAreas`
    above, and live on the same two files.
    """

    vol_blkt_inboard_full_coverage = OutputInto(fwbs)
    vol_blkt_outboard_full_coverage = OutputInto(fwbs)
    vol_blkt_total_full_coverage = OutputInto(fwbs)

    def __call__(
        self,
        r_shld_inboard_inner=From(build),
        dr_shld_inboard=From(build),
        dr_blkt_inboard=From(build),
        dr_fw_inboard=From(build),
        dr_fw_plasma_gap_inboard=From(build),
        rminor=From(physics),
        dr_fw_plasma_gap_outboard=From(build),
        dr_fw_outboard=From(build),
        dz_blkt_half=From(blanket),
        dr_blkt_outboard=From(build),
        dz_blkt_upper=From(build),
    ):
        return calculate_dshaped_blkt_volumes(
            r_shld_inboard_inner,
            dr_shld_inboard,
            dr_blkt_inboard,
            dr_fw_inboard,
            dr_fw_plasma_gap_inboard,
            rminor,
            dr_fw_plasma_gap_outboard,
            dr_fw_outboard,
            dz_blkt_half,
            dr_blkt_outboard,
            dz_blkt_upper,
        )


class BlanketCoverageFactors(ExplicitFunction):
    """The family that owns `.fwbs.vol_blkt_total` and the five fields written beside
    it: one occupant per `n_divertors` arm of `BlanketLibrary.apply_coverage_factors`.

    `.fwbs.vol_blkt_total` is what the whole of this file exists to reach:
    `CCFE_HCPB.component_masses` (`hcpb.py:306`, `:419`, `:425`, `:444`) reads it and
    nothing else in the tokamak call surface writes it.

    Both arms are written (2026-08-27); the slot is total. The arms read the same six
    fields and differ by one literal, which is enough to make them occupants rather than
    a parameter (`next_steps.md` §14.2, the `istore` precedent).
    """


class BlanketCoverageFactorsSingleNull(BlanketCoverageFactors):
    """cottax node: `apply_coverage_factors_single_null`. `n_divertors == 1`."""

    a_blkt_outboard_surface = OutputInto(build)
    a_blkt_total_surface = OutputInto(build)
    vol_blkt_outboard = OutputInto(fwbs)
    vol_blkt_inboard = OutputInto(fwbs)
    a_blkt_inboard_surface = OutputInto(build)
    vol_blkt_total = OutputInto(fwbs)

    def __call__(
        self,
        a_blkt_total_surface_full_coverage=From(build),
        a_blkt_inboard_surface_full_coverage=From(build),
        f_ster_div_single=From(fwbs),
        f_a_fw_outboard_hcd=From(fwbs),
        vol_blkt_total_full_coverage=From(fwbs),
        vol_blkt_inboard_full_coverage=From(fwbs),
    ):
        return apply_coverage_factors_single_null(
            a_blkt_total_surface_full_coverage,
            a_blkt_inboard_surface_full_coverage,
            f_ster_div_single,
            f_a_fw_outboard_hcd,
            vol_blkt_total_full_coverage,
            vol_blkt_inboard_full_coverage,
        )


class BlanketCoverageFactorsDoubleNull(BlanketCoverageFactors):
    """cottax node: `apply_coverage_factors_double_null`. `n_divertors == 2`.

    Live on `spherical_tokamak_eval.IN.DAT` and `st_regression.IN.DAT`. Carries
    PROCESS's areas-doubled/volumes-not asymmetry unrepaired -- see the function's
    docstring.
    """

    a_blkt_outboard_surface = OutputInto(build)
    a_blkt_total_surface = OutputInto(build)
    vol_blkt_outboard = OutputInto(fwbs)
    vol_blkt_inboard = OutputInto(fwbs)
    a_blkt_inboard_surface = OutputInto(build)
    vol_blkt_total = OutputInto(fwbs)

    def __call__(
        self,
        a_blkt_total_surface_full_coverage=From(build),
        a_blkt_inboard_surface_full_coverage=From(build),
        f_ster_div_single=From(fwbs),
        f_a_fw_outboard_hcd=From(fwbs),
        vol_blkt_total_full_coverage=From(fwbs),
        vol_blkt_inboard_full_coverage=From(fwbs),
    ):
        return apply_coverage_factors_double_null(
            a_blkt_total_surface_full_coverage,
            a_blkt_inboard_surface_full_coverage,
            f_ster_div_single,
            f_a_fw_outboard_hcd,
            vol_blkt_total_full_coverage,
            vol_blkt_inboard_full_coverage,
        )


class BlanketInboardPoloidalAngle(ExplicitFunction):
    """cottax node: `calculate_blkt_inboard_poloidal_plasma_angle`. Unswitched --
    `hcpb.py:64` runs it whatever `n_divertors`, `itart` or the blanket shape are, and
    the formula reads none of them.

    Owns `.blanket.deg_blkt_inboard_poloidal_plasma` only. Its immediate successor,
    `.blanket.f_deg_blkt_inboard_poloidal_plasma` (`hcpb.py:71-73`, the same angle over
    360), is UNPORTED: PROCESS writes it and only `blanket_library.py:687-688`'s
    reporting reads it, so nothing in this graph does, and owning it would add an output
    with no consumer rather than close a hole.

    **Its outboard sibling stays UNPORTED, and the reason is structural, not scope.**
    `hcpb.py:54-62` computes `.blanket.deg_blkt_outboard_poloidal_plasma` from
    `.divertor.deg_div_poloidal_plasma`, which `.tokamak.divertor.heat_flux_split` owns
    and computes *from this node's output*. So if the outboard angle is ever ported it
    must be a **separate node**: folded into this one, the merged node would read what
    the divertor writes and write what the divertor reads, and the pair would be an SCC.

    PROCESS runs the divertor (`caller.py:324`) *before* the blanket (`:343`), so its
    `Divertor.run` reads the inboard angle the **previous** pipeline pass wrote -- the
    coupling is real and `Caller.call_models`' up-to-ten-passes loop is what closes it,
    which is the implicit-cycle pattern `CLAUDE.md` describes. It stays out of this
    graph only because nothing here reads the outboard angle; that is an absence of a
    consumer, not a proof of acyclicity, and a future pass that ports it should expect
    to declare the loop rather than to find there is none.
    """

    deg_blkt_inboard_poloidal_plasma = OutputInto(blanket)

    def __call__(
        self,
        rminor=From(physics),
        dz_blkt_half=From(blanket),
        dr_fw_plasma_gap_inboard=From(build),
    ):
        return calculate_blkt_inboard_poloidal_plasma_angle(
            rminor, dz_blkt_half, dr_fw_plasma_gap_inboard
        )
