"""The four superconductor enums, vendored verbatim from
`process/models/superconductors.py`.

Not in `enums.py` because these are not bare `IntEnum`s: each member is a tuple that a
`__new__` unpacks into attached attributes (`material`, `sc_shape`, `sc_type`,
`full_name`), so the class body *is* a small material table and copying it means copying
the `__new__`/`DynamicClassAttribute` machinery that reads it. That machinery is still
a declaration -- attribute access over a fixed table, no computation -- which is why
§23.2's vendoring reaches it, but it is worth keeping visibly apart from the
generated stubs.
`indat.py:1383` reads `.sc_shape`, so the attributes are load-bearing, not decoration.

Lines 18-165 of the source, unmodified. `tests/functional_process/test_vocabulary.py`
asserts names, values *and* every attached attribute against PROCESS's.
"""

from enum import IntEnum
from types import DynamicClassAttribute


class SuperconductorShape(IntEnum):
    """Enumeration of superconductor shapes."""

    CABLE = 1
    "The superconductor is in the form of a cylindrical cable"
    TAPE = 2
    "The superconductor is in the form of a flat tape with a rectangular cross-section"


class SuperconductorType(IntEnum):
    """Enumeration of superconductor types."""

    LOW_TEMPERATURE = (1, "LTS")
    HIGH_TEMPERATURE = (2, "HTS")

    def __new__(cls, value: int, abbreviation: str):
        """Create a new instance of SuperconductorType."""
        obj = int.__new__(cls, value)
        obj._value_ = value
        obj._abbreviation_ = abbreviation
        return obj

    @DynamicClassAttribute
    def abbreviation(self):
        """The abbreviation for this superconductor type."""
        return self._abbreviation_


class SuperconductorMaterial(IntEnum):
    """Enumeration of superconductor materials."""

    NB3SN = (1, SuperconductorType.LOW_TEMPERATURE, "Nb₃Sn")
    NBTI = (2, SuperconductorType.LOW_TEMPERATURE, "NbTi")
    BI2212 = (3, SuperconductorType.HIGH_TEMPERATURE, "Bi-2212")
    REBCO = (4, SuperconductorType.HIGH_TEMPERATURE, "REBCO")

    def __new__(cls, value: int, sc_type: SuperconductorType, material_name: str):
        """Create a new instance of SuperconductorMaterial."""
        obj = int.__new__(cls, value)
        obj._value_ = value
        obj._sc_type_ = sc_type
        obj._material_name_ = material_name
        return obj

    @DynamicClassAttribute
    def sc_type(self):
        """The superconductor type (LTS or HTS) for this material."""
        return self._sc_type_.abbreviation

    @DynamicClassAttribute
    def material_name(self):
        """The name of the superconductor material."""
        return self._material_name_


class SuperconductorModel(IntEnum):
    """Enumeration of superconductor models."""

    ITER_NB3SN = (
        1,
        SuperconductorMaterial.NB3SN,
        SuperconductorShape.CABLE,
        "ITER Nb₃Sn critical surface model",
    )
    BI2212 = (2, SuperconductorMaterial.BI2212, SuperconductorShape.CABLE, "Bi-2212")
    OLD_LUBELL_NBTI = (
        3,
        SuperconductorMaterial.NBTI,
        SuperconductorShape.CABLE,
        "Old Lubell NbTi",
    )
    USER_DEFINED_NB3SN = (
        4,
        SuperconductorMaterial.NB3SN,
        SuperconductorShape.CABLE,
        "User-defined ITER Nb₃Sn",
    )
    WST_NB3SN = (
        5,
        SuperconductorMaterial.NB3SN,
        SuperconductorShape.CABLE,
        "Western Superconducting Nb₃Sn",
    )
    CROCO_REBCO = (
        6,
        SuperconductorMaterial.REBCO,
        SuperconductorShape.TAPE,
        "CROCO REBCO",
    )
    DURHAM_NBTI = (
        7,
        SuperconductorMaterial.NBTI,
        SuperconductorShape.CABLE,
        "Durham Ginzburg-Landau NbTi",
    )
    DURHAM_REBCO = (
        8,
        SuperconductorMaterial.REBCO,
        SuperconductorShape.TAPE,
        "Durham Ginzburg-Landau REBCO",
    )
    HAZELTON_ZHAI_REBCO = (
        9,
        SuperconductorMaterial.REBCO,
        SuperconductorShape.TAPE,
        "Hazelton-Zhai REBCO",
    )

    def __new__(
        cls,
        value: int,
        material: SuperconductorMaterial,
        shape: SuperconductorShape,
        full_name: str,
    ):
        """Create a new instance of SuperconductorModel."""
        obj = int.__new__(cls, value)
        obj._value_ = value
        obj._material_ = material
        obj._shape_ = shape
        obj._full_name_ = full_name
        return obj

    @DynamicClassAttribute
    def material(self):
        """The superconductor material associated with this model."""
        return self._material_

    @DynamicClassAttribute
    def material_name(self):
        """The name of the superconductor material associated with this model."""
        return self._material_.material_name

    @DynamicClassAttribute
    def sc_shape(self):
        """The superconductor shape associated with this model."""
        return self._shape_

    @DynamicClassAttribute
    def sc_type(self):
        """The superconductor type (LTS or HTS) associated with this model."""
        return self._material_.sc_type

    @DynamicClassAttribute
    def full_name(self):
        """The full name of this superconductor model."""
        return self._full_name_
