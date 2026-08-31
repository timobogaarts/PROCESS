"""Switch enums, vendored from PROCESS (§23.2).

Declarations only -- every one is a name/value table PROCESS uses to spell an integer
switch. Most carry attached attributes (`full_name`, `description`, `abbreviation`, and
`CurrentDriveModel.method`, which `indat.py` branches on), so the class *body* is the
table and `__new__`/`DynamicClassAttribute` is how it is read back. That machinery is
still a declaration -- attribute access over a fixed tuple, no computation -- so §23.2's
vendoring reaches it.

**Every class here is `inspect.getsource` of PROCESS's, unmodified**, emitted in
dependency order with the source module named above it. Generated, not retyped, for the
same reason unit #8 declined to retype the stellarator presets. The four superconductor
enums live in `superconductors.py` instead, and `constants`/`areas`/`iteration_variables`
/`stellarator_presets` are the other vendored kinds.

`tests/functional_process/test_vocabulary.py` asserts, per class, that the vendored
member names, values *and* attached attributes equal PROCESS's -- the equality test that
pays for the copy.
"""

# The classes below are copies, so their lines are PROCESS's to wrap, not this file's.
# ruff: noqa: E501

from enum import IntEnum
from types import DynamicClassAttribute


# from `process.data_structure.blanket_variables`
class BlktModelTypes(IntEnum):
    """Enum for blanket model types. `i_blanket_type`"""

    CCFE_HCPB = 1
    DCLL = 5


# from `process.data_structure.build_variables`
class TFCSRadialConfiguration(IntEnum):
    """Switch for placing the TF coil inside the CS, controlled through `BuildData.i_tf_inside_cs`"""

    TF_OUTSIDE_CS = (0, "Inboard TF coil leg is outside the CS")
    TF_INSIDE_CS = (1, "Inboard TF coil leg is inside the CS")

    def __new__(cls, value: int, description: str):
        """Create a new TFCSRadialConfiguration enum member with value and description.

        Parameters
        ----------
        value : int
            The numeric value of the enum member.
        description : str
            The description of the TFCSRadialConfiguration enum member.

        Returns
        -------
        TFCSRadialConfiguration
            A new enum member with the specified value and description.
        """
        obj = int.__new__(cls, value)
        obj._value_ = value
        obj._description_ = description
        return obj

    @DynamicClassAttribute
    def description(self):
        """Description of the TF inboard and CS coil radial configuration."""
        return self._description_


# from `process.data_structure.divertor_variables`
class DivertorHeatLoadModel(IntEnum):
    """Divertor heat load model enumeration, controlled' by `i_div_heat_load`"""

    USER_INPUT = 0
    """User input for divertor heat load"""

    PENG_CHAMBER = 1
    """Divertor heat load model based on Peng chamber"""

    WADE = 2
    """Divertor heat load model based on Wade (Wade 2020)"""


# from `process.data_structure.numerics`
class FiguresOfMerit(IntEnum):
    """Enumeration of the available figures of merit (FoM) that can be used as
    objective functions for optimisation in PROCESS.
    """

    MAJOR_RADIUS = (1, "Plasma major radius (R₀)")
    NEUTRON_WALL_LOAD = (3, "Neutron wall load")
    P_TF_PLUS_P_PF = (4, "TF & PF coil power")
    FUSION_GAIN_Q = (5, "Fusion gain (Qₚₗₐₛₘₐ)")
    COST_OF_ELECTRICITY = (6, "Cost of electricity")
    CAPITAL_COST = (7, "Plant capital cost")
    ASPECT_RATIO = (8, "Plasma aspect ratio")
    DIVERTOR_HEAT_LOAD = (9, "Divertor heat load")
    TOROIDAL_FIELD = (10, "Plasma toroidal field on axis (B₀)")
    TOTAL_INJECTED_POWER = (11, "Plasma total injected power (Pᵢₙⱼ)")
    PULSE_LENGTH = (14, "Pulse length")
    PLANT_AVAILABILITY_FACTOR = (15, "Plant availability factor")
    MIN_R0_MAX_TAU_BURN = (
        16,
        "Linear combination of major radius (minimised) and pulse length (maximised)",
    )
    NET_ELECTRICAL_OUTPUT = (17, "Plant net electrical output")
    NULL_FIGURE_OF_MERIT = (18, "Null Figure of Merit")
    MAX_Q_MAX_T_PLANT_PULSE_BURN = (
        19,
        "Linear combination of big Q and pulse length (maximised)",
    )

    def __new__(cls, value: int, description: str):
        """Create a new FiguresOfMerit enum member with description.

        Args:
            value: The integer value of the enum member.
            description: The description for this figure of merit.

        Returns
        -------
            The new enum member with attached description.
        """
        obj = int.__new__(cls, value)
        obj._value_ = value
        obj._description_ = description
        return obj

    @DynamicClassAttribute
    def description(self):
        """The description for this figure of merit."""
        return self._description_


# from `process.data_structure.pfcoil_variables`
class PFConductorModel(IntEnum):
    """Enumeration for PF conductor models.
    Controlled via `i_pf_conductor`

    """

    SUPERCONDUCTING = 0
    RESISTIVE = 1


# from `process.data_structure.physics_variables`
class ConfinementRadiationLossModel(IntEnum):
    """Confinement radiation loss model types"""

    FULL_RADIATION = (0, "All radiation included in loss power term")
    CORE_ONLY = (1, "Only core radiation included in loss power term")
    NO_RADIATION = (2, "No radiation included in loss power term")

    def __new__(cls, value: int, description: str):
        """Create a new instance of ConfinementRadiationLossModel.

        Parameters
        ----------
        value : int
            The enum value
        description : str
            The description of the radiation loss model

        Returns
        -------
        ConfinementRadiationLossModel
            A new enum instance with the given value and description
        """
        obj = int.__new__(cls, value)
        obj._value_ = value
        obj.description = description
        return obj


# from `process.data_structure.physics_variables`
class ConfinementMode(IntEnum):
    """Enum for plasma confinement mode"""

    L_MODE = (0, "L")
    H_MODE = (1, "H")
    I_MODE = (2, "I")
    STELLARATOR = (3, "Stell")
    OHMIC = (4, "Ohmic")

    def __new__(cls, value: int, abbreviation: str):
        """Create a new instance of ConfinementMode.

        Parameters
        ----------
        value : int
            The enum value
        abbreviation : str
            The abbreviation of the confinement mode

        Returns
        -------
        ConfinementMode
            A new enum instance with the given value and abbreviation
        """
        obj = int.__new__(cls, value)
        obj._value_ = value
        obj.abbreviation = abbreviation

        return obj


# from `process.data_structure.physics_variables`
class ConfinementTimeModel(IntEnum):
    """Confinement time (τ_E) model types"""

    USER_INPUT = (0, "User input electron confinement   ", None)
    NEO_ALCATOR = (
        1,
        f"Neo-Alcator                ({ConfinementMode.OHMIC.abbreviation})",
        ConfinementMode.OHMIC,
    )
    MIRNOV = (
        2,
        f"Mirnov                         ({ConfinementMode.H_MODE.abbreviation})",
        ConfinementMode.H_MODE,
    )
    MEREZHKIN_MUHKOVATOV = (
        3,
        f"Merezkhin-Muhkovatov    ({ConfinementMode.OHMIC.abbreviation})({ConfinementMode.L_MODE.abbreviation})",
        ConfinementMode.OHMIC | ConfinementMode.L_MODE,
    )
    SHIMOMURA = (
        4,
        f"Shimomura                      ({ConfinementMode.H_MODE.abbreviation})",
        ConfinementMode.H_MODE,
    )
    KAYE_GOLDSTON = (
        5,
        f"Kaye-Goldston                  ({ConfinementMode.L_MODE.abbreviation})",
        ConfinementMode.L_MODE,
    )
    ITER_89P = (
        6,
        f"ITER 89-P                      ({ConfinementMode.L_MODE.abbreviation})",
        ConfinementMode.L_MODE,
    )
    ITER_89_0 = (
        7,
        f"ITER 89-O                      ({ConfinementMode.L_MODE.abbreviation})",
        ConfinementMode.L_MODE,
    )
    REBUT_LALLIA = (
        8,
        f"Rebut-Lallia                   ({ConfinementMode.L_MODE.abbreviation})",
        ConfinementMode.L_MODE,
    )
    GOLDSTON = (
        9,
        f"Goldston                       ({ConfinementMode.L_MODE.abbreviation})",
        ConfinementMode.L_MODE,
    )
    T_10 = (
        10,
        f"T10                            ({ConfinementMode.L_MODE.abbreviation})",
        ConfinementMode.L_MODE,
    )
    JAERI = (
        11,
        f"JAERI / Odajima-Shimomura      ({ConfinementMode.L_MODE.abbreviation})",
        ConfinementMode.L_MODE,
    )
    KAYE_BIG = (
        12,
        f"Kaye-Big Complex               ({ConfinementMode.L_MODE.abbreviation})",
        ConfinementMode.L_MODE,
    )
    ITER_H90_P = (
        13,
        f"ITER H90-P                     ({ConfinementMode.H_MODE.abbreviation})",
        ConfinementMode.H_MODE,
    )
    MINIMUM_OF_ITER_89P_AND_ITER_89_0 = (
        14,
        f"ITER 89-P & 89-O min           ({ConfinementMode.L_MODE.abbreviation})",
        ConfinementMode.L_MODE,
    )
    RIEDEL_L = (
        15,
        f"Riedel                         ({ConfinementMode.L_MODE.abbreviation})",
        ConfinementMode.L_MODE,
    )
    CHRISTIANSEN = (
        16,
        f"Christiansen                   ({ConfinementMode.L_MODE.abbreviation})",
        ConfinementMode.L_MODE,
    )
    LACKNER_GOTTARDI = (
        17,
        f"Lackner-Gottardi               ({ConfinementMode.L_MODE.abbreviation})",
        ConfinementMode.L_MODE,
    )
    NEO_KAYE = (
        18,
        f"Neo-Kaye                       ({ConfinementMode.L_MODE.abbreviation})",
        ConfinementMode.L_MODE,
    )
    RIEDEL_H = (
        19,
        f"Riedel                         ({ConfinementMode.H_MODE.abbreviation})",
        ConfinementMode.H_MODE,
    )
    ITER_H90_P_AMENDED = (
        20,
        f"ITER H90-P amended             ({ConfinementMode.H_MODE.abbreviation})",
        ConfinementMode.H_MODE,
    )
    SUDO_ET_AL = (
        21,
        f"LHD                        ({ConfinementMode.STELLARATOR.abbreviation})",
        ConfinementMode.STELLARATOR,
    )
    GYRO_REDUCED_BOHM = (
        22,
        f"Gyro-reduced Bohm          ({ConfinementMode.STELLARATOR.abbreviation})",
        ConfinementMode.STELLARATOR,
    )
    LACKNER_GOTTARDI_STELLARATOR = (
        23,
        f"Lackner-Gottardi           ({ConfinementMode.STELLARATOR.abbreviation})",
        ConfinementMode.STELLARATOR,
    )
    ITER_93H = (
        24,
        f"ITER-93H  ELM-free             ({ConfinementMode.H_MODE.abbreviation})",
        ConfinementMode.H_MODE,
    )
    TITAN_REMOVED = (
        25,
        "TITAN RFP OBSOLETE                (N/A)",
        None,
    )
    ITER_H97P = (
        26,
        f"ITER H-97P ELM-free            ({ConfinementMode.H_MODE.abbreviation})",
        ConfinementMode.H_MODE,
    )
    ITER_H97P_ELMY = (
        27,
        f"ITER H-97P ELMy                ({ConfinementMode.H_MODE.abbreviation})",
        ConfinementMode.H_MODE,
    )
    ITER_96P = (
        28,
        f"ITER-96P (ITER-97L)            ({ConfinementMode.L_MODE.abbreviation})",
        ConfinementMode.L_MODE,
    )
    VALOVIC_ELMY = (
        29,
        f"Valovic modified ELMy          ({ConfinementMode.H_MODE.abbreviation})",
        ConfinementMode.H_MODE,
    )
    KAYE = (
        30,
        f"Kaye 98 modified               ({ConfinementMode.L_MODE.abbreviation})",
        ConfinementMode.L_MODE,
    )
    ITER_PB98P_Y = (
        31,
        f"ITERH-PB98P(y)                 ({ConfinementMode.H_MODE.abbreviation})",
        ConfinementMode.H_MODE,
    )
    IPB98_Y = (
        32,
        f"IPB98(y)                       ({ConfinementMode.H_MODE.abbreviation})",
        ConfinementMode.H_MODE,
    )
    ITER_IPB98Y1 = (
        33,
        f"IPB98(y,1)                     ({ConfinementMode.H_MODE.abbreviation})",
        ConfinementMode.H_MODE,
    )
    ITER_IPB98Y2 = (
        34,
        f"IPB98(y,2)                     ({ConfinementMode.H_MODE.abbreviation})",
        ConfinementMode.H_MODE,
    )
    ITER_IPB98Y3 = (
        35,
        f"IPB98(y,3)                     ({ConfinementMode.H_MODE.abbreviation})",
        ConfinementMode.H_MODE,
    )
    ITER_IPB98Y4 = (
        36,
        f"IPB98(y,4)                     ({ConfinementMode.H_MODE.abbreviation})",
        ConfinementMode.H_MODE,
    )
    ISS95_STELLARATOR = (
        37,
        f"ISS95                      ({ConfinementMode.STELLARATOR.abbreviation})",
        ConfinementMode.STELLARATOR,
    )
    ISS04_STELLARATOR = (
        38,
        f"ISS04                      ({ConfinementMode.STELLARATOR.abbreviation})",
        ConfinementMode.STELLARATOR,
    )
    DS03 = (
        39,
        f"DS03 beta-independent          ({ConfinementMode.H_MODE.abbreviation})",
        ConfinementMode.H_MODE,
    )
    MURARI = (
        40,
        f'Murari "Non-power law"         ({ConfinementMode.H_MODE.abbreviation})',
        ConfinementMode.H_MODE,
    )
    PETTY08 = (
        41,
        f"Petty 2008                     ({ConfinementMode.H_MODE.abbreviation})",
        ConfinementMode.H_MODE,
    )
    LANG_HIGH_DENSITY = (
        42,
        f"Lang high density              ({ConfinementMode.H_MODE.abbreviation})",
        ConfinementMode.H_MODE,
    )
    HUBBARD_NOMINAL = (
        43,
        f"Hubbard 2017 - nominal         ({ConfinementMode.I_MODE.abbreviation})",
        ConfinementMode.I_MODE,
    )
    HUBBARD_LOWER = (
        44,
        f"Hubbard 2017 - lower           ({ConfinementMode.I_MODE.abbreviation})",
        ConfinementMode.I_MODE,
    )
    HUBBARD_UPPER = (
        45,
        f"Hubbard 2017 - upper           ({ConfinementMode.I_MODE.abbreviation})",
        ConfinementMode.I_MODE,
    )
    MENARD_NSTX = (
        46,
        f"Menard NSTX                    ({ConfinementMode.H_MODE.abbreviation})",
        ConfinementMode.H_MODE,
    )
    MENARD_NSTX_PETTY08_HYBRID = (
        47,
        f"Menard NSTX-Petty08 hybrid     ({ConfinementMode.H_MODE.abbreviation})",
        ConfinementMode.H_MODE,
    )
    NSTX_GYRO_BOHM = (
        48,
        f"Buxton NSTX gyro-Bohm          ({ConfinementMode.H_MODE.abbreviation})",
        ConfinementMode.H_MODE,
    )
    ITPA20 = (
        49,
        f"ITPA20                         ({ConfinementMode.H_MODE.abbreviation})",
        ConfinementMode.H_MODE,
    )
    ITPA20_IL = (
        50,
        f"ITPA20-IL                      ({ConfinementMode.H_MODE.abbreviation})",
        ConfinementMode.H_MODE,
    )
    NCST = (
        51,
        f"NCST                           ({ConfinementMode.L_MODE.abbreviation})",
        ConfinementMode.L_MODE,
    )
    PAZ_SOLDAN_NT = (
        51,
        f"Paz-Soldan Neg Triang          ({ConfinementMode.L_MODE.abbreviation})",
        ConfinementMode.L_MODE,
    )

    def __new__(cls, value: int, full_name: str, mode: ConfinementMode = None):
        """Create a new instance of ConfinementTimeModel.

        Parameters
        ----------
        value : int
            The enum value
        full_name : str
            The full name of the confinement time model
        mode : ConfinementMode
            The confinement mode associated with the model

        Returns
        -------
        ConfinementTimeModel
            A new enum instance with the given value and full name
        """
        obj = int.__new__(cls, value)
        obj._value_ = value
        obj.full_name = full_name
        obj.mode = mode
        return obj


# from `process.data_structure.physics_variables`
class CurrentProfileIndexModel(IntEnum):
    """Enum for current profile index models."""

    USER_INPUT = 0
    WESSON = 1


# from `process.data_structure.physics_variables`
class DivertorNumberModels(IntEnum):
    """Enum for divertor number models. `i_single_null` is the index for this enum."""

    DOUBLE_NULL = 0
    SINGLE_NULL = 1


# from `process.data_structure.physics_variables`
class OutbordSOLPowerDecayLengthModel(IntEnum):
    """Enum for outboard scrape off layer power decay length models with descriptions."""

    USER_INPUT = (0, "User input")
    EICH_2013 = (1, "Eich 2013")
    MAST_2014_1 = (2, "MAST 2014-1")
    MAST_2014_2 = (3, "MAST 2014-2")

    def __new__(cls, value: int, description: str):
        """Create a new instance of OutbordSOLPowerDecayLengthModel.

        Parameters
        ----------
        value : int
            The enum value
        description : str
            The description of the model

        Returns
        -------
        OutbordSOLPowerDecayLengthModel
            A new enum instance with the given value and description
        """
        obj = int.__new__(cls, value)
        obj._value_ = value
        obj.description = description
        return obj


# from `process.data_structure.physics_variables`
class PlasmaIgnitionModel(IntEnum):
    """Enum for plasma ignition models."""

    NON_IGNITED = 0
    IGNITED = 1


# from `process.data_structure.superconducting_tf_coil_variables`
class TFWPIntegerTurnType(IntEnum):
    """TF winding pack integer turn type, controlled via the `i_tf_turns_integer`
    variable.
    """

    NON_INTEGER = 0
    """Non integer number of turns in TF winding pack"""

    INTEGER = 1
    """Integer number of turns in TF winding pack"""


# from `process.models.availability`
class AvailabilityModel(IntEnum):
    """Enum for availability models"""

    USER_INPUT = (0, "Input value for `f_t_plant_available`")
    WARD_TAYLOR = (1, "Ward and Taylor model (1999)")
    MORRIS = (2, "Morris model (2015)")
    ST = (3, "ST model (2023)")

    def __new__(cls, value: int, full_name: str):
        """Create a new AvailabilityModel enum instance.

        Parameters
        ----------
        value : int
            The integer value of the enum member.
        full_name : str
            The full name/description of the availability model.

        Returns
        -------
        AvailabilityModel
            A new enum instance with the specified value and full_name.
        """
        obj = int.__new__(cls, value)
        obj._value_ = value
        obj._full_name_ = full_name
        return obj

    @DynamicClassAttribute
    def full_name(self):
        """The full name of the availability model."""
        return self._full_name_


# from `process.models.build`
class FwBlktVVShape(IntEnum):
    """Enum for first wall, blanket, and vacuum vessel shape options."""

    D_SHAPED = 1
    ELLIPTICAL_SHAPED = 2


# from `process.models.physics.bootstrap_current`
class BootstrapCurrentFractionModel(IntEnum):
    """Bootstrap plasma current fraction (f_BS) model types"""

    USER_INPUT = (0, "User Input")
    ITER_89 = (1, "ITER IPDG89 scaling")
    NEVINS = (2, "Nevins scaling")
    WILSON = (3, "Wilson scaling")
    SAUTER = (4, "Sauter scaling")
    SAKAI = (5, "Sakai scaling")
    ARIES = (6, "Aries scaling")
    ANDRADE = (7, "Andrade scaling")
    HOANG = (8, "Hoang scaling")
    WONG = (9, "Wong scaling")
    GI_1 = (10, "GI 1 scaling")
    GI_2 = (11, "GI 2 scaling")
    SUGIYAMA_L_MODE = (12, "Sugiyama L Mode scaling")
    SUGIYAMA_H_MODE = (13, "Sugiyama H Mode scaling")

    def __new__(cls, value: int, full_name: str):
        """Create a new BootstrapCurrentFractionModel enum instance.

        Parameters
        ----------
        value : int
            The integer value of the enum member.
        full_name : str
            The full name/description of the bootstrap current model.

        Returns
        -------
        BootstrapCurrentFractionModel
            A new enum instance with the specified value and full_name.
        """
        obj = int.__new__(cls, value)
        obj._value_ = value
        obj._full_name_ = full_name
        return obj

    @DynamicClassAttribute
    def full_name(self):
        """The full name of the bootstrap current model."""
        return self._full_name_


# from `process.models.physics.current_drive`
class CurrentDriveMethodType(IntEnum):
    """Enum for heating and current drive method types"""

    NONE = (0, "None")
    LOWER_HYBRID = (1, "LHCD")
    ION_CYCLOTRON = (2, "ICCD")
    ELECTRON_CYCLOTRON = (3, "ECRH")
    NEUTRAL_BEAM = (4, "NBI")
    ELECTRON_BERNSTEIN = (5, "EBW")

    def __new__(cls, value: int, abbreviation: str):
        """Create a new CurrentDriveMethodType enum member."""
        obj = int.__new__(cls, value)
        obj._value_ = value
        obj._abbreviation_ = abbreviation
        return obj

    @DynamicClassAttribute
    def abbreviation(self):
        """The abbreviation for this current drive method type."""
        return self._abbreviation_


# from `process.models.physics.current_drive`
class CurrentDriveModel(IntEnum):
    """Heating and current drive models for use in current drive calculations"""

    NO_CURRENT_DRIVE = (
        0,
        CurrentDriveMethodType.NONE,
        "No Current Drive",
    )

    FENSTERMACHER_LOWER_HYBRID = (
        1,
        CurrentDriveMethodType.LOWER_HYBRID,
        "Fenstermacher Lower Hybrid",
    )
    IPDG89_ION_CYCLOTRON = (
        2,
        CurrentDriveMethodType.ION_CYCLOTRON,
        "IPDG89 Ion Cyclotron",
    )
    FENSTERMACHER_ELECTRON_CYCLOTRON = (
        3,
        CurrentDriveMethodType.ELECTRON_CYCLOTRON,
        "Fenstermacher Electron Cyclotron",
    )
    EHST_LOWER_HYBRID = (
        4,
        CurrentDriveMethodType.LOWER_HYBRID,
        "EHST Lower Hybrid",
    )
    ITER_NEUTRAL_BEAM = (
        5,
        CurrentDriveMethodType.NEUTRAL_BEAM,
        "ITER Neutral Beam",
    )
    CULHAM_LOWER_HYBRID = (
        6,
        CurrentDriveMethodType.LOWER_HYBRID,
        "Culham Lower Hybrid",
    )
    CULHAM_ELECTRON_CYCLOTRON = (
        7,
        CurrentDriveMethodType.ELECTRON_CYCLOTRON,
        "Culham Electron Cyclotron",
    )
    CULHAM_NEUTRAL_BEAM = (
        8,
        CurrentDriveMethodType.NEUTRAL_BEAM,
        "Culham Neutral Beam",
    )
    USER_INPUT_ELECTRON_CYCLOTRON = (
        10,
        CurrentDriveMethodType.ELECTRON_CYCLOTRON,
        "User Input Electron Cyclotron",
    )
    USER_INPUT_ELECTRON_BERNSTEIN = (
        12,
        CurrentDriveMethodType.ELECTRON_BERNSTEIN,
        "User Input Electron Bernstein",
    )
    FREETHY_ELECTRON_CYCLOTRON = (
        13,
        CurrentDriveMethodType.ELECTRON_CYCLOTRON,
        "Freethy Electron Cyclotron",
    )

    def __new__(cls, value: int, method: CurrentDriveMethodType, full_name: str):
        """Create a new CurrentDriveModel enum member."""
        obj = int.__new__(cls, value)
        obj._value_ = value
        obj._method_ = method
        obj._full_name_ = full_name
        return obj

    @DynamicClassAttribute
    def method(self):
        """The current drive method type for this current drive model."""
        return self._method_

    @DynamicClassAttribute
    def abbreviation(self):
        """The abbreviation for this current drive model."""
        return self.method.abbreviation

    @DynamicClassAttribute
    def full_name(self):
        """The full name for this current drive model."""
        return self._full_name_


# from `process.models.physics.density_limit`
class DensityLimitModel(IntEnum):
    """Electron density model types"""

    ASDEX = (1, "ASDEX limit")
    BORRASS_ITER_I = (2, "Borrass ITER I limit")
    BORRASS_ITER_II = (3, "Borrass ITER II limit")
    JET_EDGE_RADIATION = (4, "JET Edge Radiation limit")
    JET_SIMPLE = (5, "JET Simple limit")
    HUGILL_MURAKAMI = (6, "Hugill Murakami limit")
    GREENWALD = (7, "Greenwald limit")
    ASDEX_NEW = (8, "ASDEX New limit")

    def __new__(cls, value: int, full_name: str):
        """Create a new DensityLimitModel instance.

        Parameters
        ----------
        value : int
            The integer value for the enum member.
        full_name : str
            The full descriptive name of the density limit model.

        Returns
        -------
        obj
            A new instance of DensityLimitModel.
        """
        obj = int.__new__(cls, value)
        obj._value_ = value
        obj._full_name_ = full_name
        return obj

    @DynamicClassAttribute
    def full_name(self):
        """The full name of the density limit model."""
        return self._full_name_


# from `process.models.physics.l_h_transition`
class PlasmaConfinementTransitionModel(IntEnum):
    """Enum for plasma L -> H and L -> I transition power threshold models."""

    ITER1996_NOMINAL = (1, "ITER-1996 Nominal")
    ITER1996_UPPER = (2, "ITER-1996 Upper")
    ITER1996_LOWER = (3, "ITER-1996 Lower")
    SNIPES1997_ITER = (4, "Snipes 1997 ITER Scaling I")
    SNIPES1997_KAPPA = (5, "Snipes 1997 ITER Scaling II")
    MARTIN08_NOMINAL = (6, "Martin 2008 Nominal")
    MARTIN08_UPPER = (7, "Martin 2008 Upper")
    MARTIN08_LOWER = (8, "Martin 2008 Lower")
    SNIPES2000_NOMINAL = (9, "Snipes 2000 Nominal")
    SNIPES2000_UPPER = (10, "Snipes 2000 Upper")
    SNIPES2000_LOWER = (11, "Snipes 2000 Lower")
    SNIPES2000_CLOSED_DIVERTOR_NOMINAL = (12, "Snipes 2000 Closed Divertor Nominal")
    SNIPES2000_CLOSED_DIVERTOR_UPPER = (13, "Snipes 2000 Closed Divertor Upper")
    SNIPES2000_CLOSED_DIVERTOR_LOWER = (14, "Snipes 2000 Closed Divertor Lower")
    HUBBARD2012_NOMINAL = (15, "Hubbard 2012 Nominal")
    HUBBARD2012_LOWER = (16, "Hubbard 2012 Lower")
    HUBBARD2012_UPPER = (17, "Hubbard 2012 Upper")
    HUBBARD2017_I_MODE = (18, "Hubbard 2017 I-Mode")
    MARTIN08_ASPECT_NOMINAL = (19, "Martin 2008 Aspect Corrected Nominal")
    MARTIN08_ASPECT_UPPER = (20, "Martin 2008 Aspect Corrected Upper")
    MARTIN08_ASPECT_LOWER = (21, "Martin 2008 Aspect Corrected Lower")

    def __new__(cls, value: int, full_name: str):
        """Create a new PlasmaConfinementTransitionModel instance.

        Parameters
        ----------
        value : int
            The integer value of the enum member.
        full_name : str
            The full descriptive name of the enum member.

        Returns
        -------
        PlasmaConfinementTransitionModel
            A new instance of PlasmaConfinementTransitionModel.
        """
        obj = int.__new__(cls, value)
        obj._value_ = value
        obj.full_name = full_name
        return obj


# from `process.models.physics.physics`
class BetaComponentLimits(IntEnum):
    """Beta component to apply limit types"""

    TOTAL = (0, "Total Beta")
    THERMAL = (1, "Thermal Beta")
    THERMAL_AND_BEAM = (2, "Thermal and Beam Beta")
    TOROIDAL = (3, "Toroidal Beta")

    def __new__(cls, value: int, full_name: str):
        """Create a new instance of BetaComponentLimits."""
        obj = int.__new__(cls, value)
        obj._value_ = value
        obj._full_name_ = full_name
        return obj

    @DynamicClassAttribute
    def full_name(self):
        """The full name of the beta component limit."""
        return self._full_name_


# from `process.models.physics.physics`
class IndInternalNormModel(IntEnum):
    """Normalised internal inductance (lᵢ) model types"""

    USER_INPUT = (0, "User Input")
    WESSON = (1, "Wesson scaling")
    MENARD = (2, "Menard scaling")

    def __new__(cls, value: int, full_name: str):
        """Create a new instance of the IndInternalNormModel enum."""
        obj = int.__new__(cls, value)
        obj._value_ = value
        obj._full_name_ = full_name
        return obj

    @DynamicClassAttribute
    def full_name(self):
        """The full name of the normalised internal inductance model."""
        return self._full_name_


# from `process.models.physics.plasma_current`
class PlasmaCurrentModel(IntEnum):
    """Enumeration of plasma current scaling models available for calculations.

    Each model represents a different scaling law used to calculate plasma
    current based on various plasma and machine parameters.
    """

    PENG_ANALYTIC_FIT = (1, "Peng analytic fit")
    PENG_DIVERTOR_SCALING = (2, "Peng divertor scaling")
    ITER_SCALING = (3, "Simple ITER scaling (cylindrical case)")
    IPDG89_SCALING = (4, "IPDG89 scaling")
    TODD_EMPIRICAL_SCALING_I = (5, "Todd empirical scaling I")
    TODD_EMPIRICAL_SCALING_II = (6, "Todd empirical scaling II")
    CONNOR_HASTIE_MODEL = (7, "Connor-Hastie model")
    SAUTER_SCALING = (8, "Sauter scaling")
    FIESTA_ST_SCALING = (9, "FIESTA ST scaling")

    def __new__(cls, value: int, full_name: str):
        """Create a new PlasmaCurrentModel enum member with value and full_name.

        Parameters
        ----------
        value : int
            The numeric value of the enum member.
        full_name : str
            The full name description of the plasma current model.

        Returns
        -------
        PlasmaCurrentModel
            A new enum member with the specified value and full_name.
        """
        obj = int.__new__(cls, value)
        obj._value_ = value
        obj._full_name_ = full_name
        return obj

    @DynamicClassAttribute
    def full_name(self):
        """The full name of the plasma current model."""
        return self._full_name_


# from `process.models.physics.plasma_current`
class PlasmaDiamagneticCurrentModel(IntEnum):
    """Enum for plasma diamagnetic current method types"""

    NONE = (0, "None")
    HENDER_ST_FIT = (1, "Hender ST fit")
    SCENE_FIT = (2, "SCENE fit")

    def __new__(cls, value: int, full_name: str):
        """Create a new enum member with value and full name.

        Parameters
        ----------
        value :
            The enum value
        full_name :
            The full name description of the enum member

        Returns
        -------
        PlasmaDiamagneticCurrentModel
            The new enum member
        """
        obj = int.__new__(cls, value)
        obj._value_ = value
        obj._full_name_ = full_name
        return obj

    @DynamicClassAttribute
    def full_name(self):
        """The full name of the enum member.

        Returns
        -------
        str
            The full name description
        """
        return self._full_name_


# from `process.models.physics.plasma_geometry`
class PlasmaGeometryModels(IntEnum):
    """Enum for plasma geometry model types."""

    USER_INPUT = (0, "User Input")
    IPDG89 = (1, "IPDG89")
    STAR_CODE = (2, "STAR Code")
    ZOHM_ITER = (3, "Zohm ITER Scaling")
    MAST_DATA = (4, "Fit to MAST data")
    FIESTA_RUNS = (5, "Fiesta Runs")
    CREATE_DATA_EU_DEMO = (6, "CREATE Data EU DEMO")
    MENARD_2016 = (7, "Menard 2016 ST Scaling")
    UNKNOWN = (8, "Unknown")
    MENARD_1997 = (9, "Menard 1997 ST Scaling")

    def __new__(cls, value: int, description: str):
        """Create a new PlasmaGeometryModels instance."""
        obj = int.__new__(cls, value)
        obj._value_ = value
        obj._description_ = description
        return obj

    @DynamicClassAttribute
    def description(self):
        """The description of the plasma geometry model."""
        return self._description_


# from `process.models.physics.plasma_geometry`
class PlasmaGeometryModelType(IntEnum):
    """Enum for i_plasma_geometry plasma geometry model types."""

    IPDG89_X_POINT = (
        0,
        PlasmaGeometryModels.USER_INPUT,
        PlasmaGeometryModels.USER_INPUT,
        PlasmaGeometryModels.IPDG89,
        PlasmaGeometryModels.IPDG89,
    )
    STAR_FIESTA = (
        1,
        PlasmaGeometryModels.STAR_CODE,
        PlasmaGeometryModels.STAR_CODE,
        PlasmaGeometryModels.FIESTA_RUNS,
        PlasmaGeometryModels.FIESTA_RUNS,
    )
    ZOHM_ITER_X_POINT = (
        2,
        PlasmaGeometryModels.ZOHM_ITER,
        PlasmaGeometryModels.USER_INPUT,
        PlasmaGeometryModels.IPDG89,
        PlasmaGeometryModels.IPDG89,
    )
    ZOHM_ITER_95 = (
        3,
        PlasmaGeometryModels.ZOHM_ITER,
        PlasmaGeometryModels.IPDG89,
        PlasmaGeometryModels.IPDG89,
        PlasmaGeometryModels.USER_INPUT,
    )
    IPDG89_95 = (
        4,
        PlasmaGeometryModels.IPDG89,
        PlasmaGeometryModels.IPDG89,
        PlasmaGeometryModels.USER_INPUT,
        PlasmaGeometryModels.USER_INPUT,
    )
    MAST_DATA_95 = (
        5,
        PlasmaGeometryModels.MAST_DATA,
        PlasmaGeometryModels.MAST_DATA,
        PlasmaGeometryModels.USER_INPUT,
        PlasmaGeometryModels.USER_INPUT,
    )
    MAST_DATA_X_POINT = (
        6,
        PlasmaGeometryModels.USER_INPUT,
        PlasmaGeometryModels.USER_INPUT,
        PlasmaGeometryModels.MAST_DATA,
        PlasmaGeometryModels.MAST_DATA,
    )
    FIESTA_RUNS_95 = (
        7,
        PlasmaGeometryModels.FIESTA_RUNS,
        PlasmaGeometryModels.FIESTA_RUNS,
        PlasmaGeometryModels.USER_INPUT,
        PlasmaGeometryModels.USER_INPUT,
    )
    FIESTA_RUNS_X_POINT = (
        8,
        PlasmaGeometryModels.USER_INPUT,
        PlasmaGeometryModels.USER_INPUT,
        PlasmaGeometryModels.FIESTA_RUNS,
        PlasmaGeometryModels.FIESTA_RUNS,
    )
    INDUCTANCE_SCALING_X_POINT = (
        9,
        PlasmaGeometryModels.UNKNOWN,
        PlasmaGeometryModels.USER_INPUT,
        PlasmaGeometryModels.IPDG89,
        PlasmaGeometryModels.IPDG89,
    )
    CREATE_DATA_EU_DEMO_X_POINT = (
        10,
        PlasmaGeometryModels.IPDG89,
        PlasmaGeometryModels.USER_INPUT,
        PlasmaGeometryModels.CREATE_DATA_EU_DEMO,
        PlasmaGeometryModels.IPDG89,
    )
    MENARD_2016_X_POINT = (
        11,
        PlasmaGeometryModels.MENARD_2016,
        PlasmaGeometryModels.USER_INPUT,
        PlasmaGeometryModels.IPDG89,
        PlasmaGeometryModels.IPDG89,
    )
    MENARD_1997_X_POINT = (
        12,
        PlasmaGeometryModels.MENARD_1997,
        PlasmaGeometryModels.USER_INPUT,
        PlasmaGeometryModels.IPDG89,
        PlasmaGeometryModels.IPDG89,
    )

    def __new__(
        cls,
        value: int,
        kappa_model: PlasmaGeometryModels,
        triang_model: PlasmaGeometryModels,
        kappa95_model: PlasmaGeometryModels,
        triang95_model: PlasmaGeometryModels,
    ):
        """Create a new PlasmaGeometryModelType instance."""
        obj = int.__new__(cls, value)
        obj._value_ = value
        obj._kappa_model_ = kappa_model
        obj._triang_model_ = triang_model
        obj._kappa95_model_ = kappa95_model
        obj._triang95_model_ = triang95_model
        return obj

    @DynamicClassAttribute
    def kappa_model(self):
        """The kappa model."""
        return self._kappa_model_

    @DynamicClassAttribute
    def triang_model(self):
        """The triangularity model."""
        return self._triang_model_

    @DynamicClassAttribute
    def kappa95_model(self):
        """The kappa95 model."""
        return self._kappa95_model_

    @DynamicClassAttribute
    def triang95_model(self):
        """The triangularity95 model."""
        return self._triang95_model_


# from `process.models.physics.plasma_geometry`
class PlasmaShapeModelType(IntEnum):
    """Enum for plasma shape model types."""

    PROCESS_ORIGINAL = (0, "PROCESS Original Double Arc")
    SAUTER = (1, "Sauter")

    def __new__(cls, value: int, full_name: str):
        """Create a new PlasmaShapeModelType instance."""
        obj = int.__new__(cls, value)
        obj._value_ = value
        obj._full_name_ = full_name
        return obj

    @DynamicClassAttribute
    def full_name(self):
        """The full name of the plasma shape model."""
        return self._full_name_


# from `process.models.power`
class ElectricConversionModelTypes(IntEnum):
    """Enum for thermal to electric power conversion model types."""

    CCFE_HCPB_VALUE = 0
    CCFE_HCPB_VALUE_WITH_DIVERTOR = 1
    USER_INPUT = 2
    STEAM_RANKINE_CYCLE = 3
    SUPERCRITICAL_CO2_BRAYTON_CYCLE = 4


# from `process.models.power`
class PumpingPowerModelTypes(IntEnum):
    """Pumping power model types for `i_p_coolant_pumping` in `fwbs_variables`"""

    USER_INPUT = 0
    FRACTION_OF_HEAT = 1
    MECHANICAL = 2
    MECHANICAL_WITH_PRESSURE_DROP = 3


# from `process.models.tfcoil.base`
class TFCoilShapeModel(IntEnum):
    """Enumeration for TF coil shape models.
    0: Auto-select
    1: D-shape
    2: Picture frame coil

    """

    DEFAULT = 0
    D_SHAPE = 1
    PICTURE_FRAME = 2


# from `process.models.tfcoil.base`
class TFConductorModel(IntEnum):
    """Enumeration for TF conductor models.

    0: Water cooled copper (GLIDCOP AL-15)
    1: Superconducting coil (SC)
    2: Helium cooled aluminium

    """

    WATER_COOLED_COPPER = 0
    SUPERCONDUCTING = 1
    HELIUM_COOLED_ALUMINIUM = 2


# from `process.models.tfcoil.base`
class TFPlasmaCaseType(IntEnum):
    """Enumeration for TF plasma-facing case types (i_tf_case_geom).

    0: Circular plasma facing front case
    1: Straight plasma facing front case

    """

    CIRCULAR = (0, "Circular edge plasma-facing front case")
    STRAIGHT = (1, "Straight edge plasma-facing front case")

    def __new__(cls, value: int, description: str):
        """Create a new instance of TFPlasmaCaseType with a description."""
        obj = int.__new__(cls, value)
        obj._value_ = value
        obj._description_ = description
        return obj

    @DynamicClassAttribute
    def description(self):
        """The description of the plasma-facing case type."""
        return self._description_


# from `process.models.tfcoil.superconducting`
class SuperconductingTFTurnType(IntEnum):
    """Enum for the type of TF coil turn, which determines the superconductor properties
    and stress calculations.
    """

    CABLE_IN_CONDUIT = (1, "CICC", "Cable-in-Conduit Conductor")
    CROSS_CONDUCTOR = (2, "CroCo", "Cross Conductor")

    def __new__(cls, value: int, abbreviation: str, full_name: str):
        """Create a new SuperconductingTFTurnType enum member
        with abbreviation and full name.

        Args:
            value: The integer value of the enum member.
            abbreviation: The abbreviation for this turn type.
            full_name: The full name for this turn type.

        Returns
        -------
            The new enum member with attached abbreviation and full name.
        """
        obj = int.__new__(cls, value)
        obj._value_ = value
        obj._abbreviation_ = abbreviation
        obj._full_name_ = full_name
        return obj

    @DynamicClassAttribute
    def abbreviation(self):
        """The abbreviation for this superconductor type."""
        return self._abbreviation_

    @DynamicClassAttribute
    def full_name(self):
        """The full name for this superconductor type."""
        return self._full_name_

    @classmethod
    def _missing_(cls, value):
        try:
            return cls[value]
        except KeyError:
            raise ValueError(
                f"Unsupported superconducting TF turn type: {value}"
            ) from None


# from `process.models.tfcoil.superconducting`
class SuperconductingTFWPShapeType(IntEnum):
    """Enum for the type of TF coil WP shape, which determines the geometry of the
    winding pack and ground insulation.
    """

    UNSET = -1
    RECTANGULAR = 0
    DOUBLE_RECTANGULAR = 1
    TRAPEZOIDAL = 2

    @DynamicClassAttribute
    def full_name(self):
        """The full name for this WP geometry type."""
        return self.name.title().replace("_", " ")
