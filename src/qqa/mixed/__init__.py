"""Mixed binary/integer/real modelling API."""

from qqa.mixed.augmented_lagrangian import (
    AdaptiveAugmentedLagrangian,
    ConstraintArchive,
)
from qqa.mixed.encoding import (
    IntegerEncodingPlan,
    choose_integer_encoding,
    decode_integer,
    encode_integer,
)
from qqa.mixed.problem import Constraint, MixedProblem
from qqa.mixed.relaxation import MixedRelaxation
from qqa.mixed.repair import RepairResult, repair_mixed_solution
from qqa.mixed.solve import solve_mixed
from qqa.mixed.variables import (
    Binary,
    BinaryVariable,
    Integer,
    IntegerVariable,
    Real,
    RealVariable,
    VariableKind,
    VariableSpace,
    VariableSpec,
)

__all__ = [
    "Binary",
    "BinaryVariable",
    "AdaptiveAugmentedLagrangian",
    "Constraint",
    "ConstraintArchive",
    "Integer",
    "IntegerVariable",
    "IntegerEncodingPlan",
    "MixedProblem",
    "MixedRelaxation",
    "Real",
    "RealVariable",
    "RepairResult",
    "VariableKind",
    "VariableSpace",
    "VariableSpec",
    "solve_mixed",
    "choose_integer_encoding",
    "decode_integer",
    "encode_integer",
    "repair_mixed_solution",
]
