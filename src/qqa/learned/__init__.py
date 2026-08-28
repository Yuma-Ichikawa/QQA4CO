"""Opt-in learned and adaptive helpers; the default solver remains pure QQA."""

from qqa.learned.diffusion import DiscreteDiffusionGenerator, DiscreteDiffusionResult
from qqa.learned.factor_graph import FactorGraphData, model_to_factor_graph
from qqa.learned.policy import (
    ConfidenceGatedPlanner,
    GatedDecision,
    OODGate,
    PlannerModelCard,
)
from qqa.learned.selector import OnlineSolverSelector, model_features
from qqa.learned.warmstart import FactorGraphWarmStart, factor_graph_warm_start

__all__ = [
    "DiscreteDiffusionGenerator",
    "DiscreteDiffusionResult",
    "FactorGraphData",
    "FactorGraphWarmStart",
    "ConfidenceGatedPlanner",
    "GatedDecision",
    "OODGate",
    "OnlineSolverSelector",
    "PlannerModelCard",
    "factor_graph_warm_start",
    "model_features",
    "model_to_factor_graph",
]
