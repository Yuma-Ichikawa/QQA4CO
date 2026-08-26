"""Continuous completion and elastic feasibility restoration through SCIP."""

from __future__ import annotations

import math
from collections.abc import Sequence
from contextlib import suppress
from dataclasses import dataclass
from time import perf_counter

import numpy as np

from qqa.algebraic import AlgebraicModel


@dataclass(frozen=True, slots=True)
class CompletionResult:
    feasible: bool
    accepted: bool
    improved_incumbent: bool
    status: str
    objective: float | None
    values: np.ndarray | None
    runtime: float
    fixed_variables: int


def create_completion_template(model):
    """Create an independent original-problem copy before SCIP starts solving.

    Keeping an idle template avoids copying the actively solving transformed
    problem from inside a Python heuristic callback.  Besides being portable
    across SCIP versions, this preserves the original variable names needed
    for safe postsolve injection.
    """
    try:
        from pyscipopt import Model
    except (ImportError, OSError) as exc:  # pragma: no cover - optional dependency
        raise ImportError("Continuous completion requires `qqa[scip]`.") from exc
    # The template outlives individual completion sub-SCIPs and is released
    # independently from the main model.  A thread-safe SCIP copy owns its
    # constraint data instead of sharing mutable/freeable internals with the
    # source model.  This matters for nonlinear QPLIB constraints: shared
    # copy data can otherwise be released twice during independent teardown.
    return Model(sourceModel=model, origcopy=True, globalcopy=True, threadsafe=True)


def complete_integer_assignment(
    template,
    variable_names: Sequence[str],
    values: Sequence[float],
    *,
    main_model=None,
    heuristic=None,
    algebraic: AlgebraicModel | None = None,
    time_limit: float = 1.0,
    node_limit: int = 500,
    seed: int = 0,
    verbose: bool = False,
) -> CompletionResult:
    """Fix an integer proposal in an independent sub-SCIP and complete it.

    ``template`` must be an idle original-problem copy, normally produced by
    :func:`create_completion_template` before the main solve begins.  When a
    main model is supplied, the full original-space solution is submitted via
    ``trySol`` so SCIP remains responsible for feasibility and acceptance.
    """
    if len(variable_names) != len(values):
        raise ValueError("variable_names and values must have the same length.")
    if len(set(variable_names)) != len(variable_names):
        raise ValueError("variable_names must be unique.")
    if not math.isfinite(time_limit) or time_limit <= 0:
        raise ValueError("time_limit must be finite and > 0.")
    if isinstance(node_limit, bool) or not isinstance(node_limit, int) or node_limit < 1:
        raise ValueError("node_limit must be a positive integer.")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer.")
    started = perf_counter()
    try:
        from pyscipopt import SCIP_PARAMSETTING, Model

        sub_model = Model(
            sourceModel=template,
            origcopy=True,
            globalcopy=True,
            threadsafe=True,
        )
        by_name = {variable.name: variable for variable in sub_model.getVars(transformed=False)}
        if any(name not in by_name for name in variable_names):
            return CompletionResult(
                False,
                False,
                False,
                "mapping_failed",
                None,
                None,
                perf_counter() - started,
                len(variable_names),
            )
        for name, value in zip(variable_names, values, strict=True):
            fixed = float(value)
            sub_model.chgVarLbGlobal(by_name[name], fixed)
            sub_model.chgVarUbGlobal(by_name[name], fixed)
    except Exception:
        return CompletionResult(
            False,
            False,
            False,
            "copy_failed",
            None,
            None,
            perf_counter() - started,
            len(variable_names),
        )

    if not verbose:
        sub_model.hideOutput()
    sub_model.setRealParam("limits/time", float(time_limit))
    sub_model.setLongintParam("limits/nodes", int(node_limit))
    with suppress(Exception):
        sub_model.setIntParam("parallel/maxnthreads", 1)
    with suppress(Exception):
        sub_model.setIntParam("lp/threads", 1)
    with suppress(Exception):
        sub_model.setIntParam("randomization/randomseedshift", seed)
    with suppress(Exception):
        sub_model.setIntParam("randomization/permutationseed", seed)
    with suppress(Exception):
        sub_model.setIntParam("randomization/lpseed", seed)
    with suppress(Exception):
        sub_model.setHeuristics(SCIP_PARAMSETTING.FAST)
    sub_model.optimize()
    status = str(sub_model.getStatus())
    best = sub_model.getBestSol()
    if best is None:
        return CompletionResult(
            False,
            False,
            False,
            status,
            None,
            None,
            perf_counter() - started,
            len(variable_names),
        )

    sub_variables = tuple(sub_model.getVars(transformed=False))
    solution_values = np.asarray(
        [sub_model.getSolVal(best, variable) for variable in sub_variables],
        dtype=np.float64,
    )
    by_solution_name = {
        variable.name: float(value)
        for variable, value in zip(sub_variables, solution_values, strict=True)
    }
    algebraic_objective = None
    objective_auxiliary = None
    if algebraic is not None and all(name in by_solution_name for name in algebraic.variable_names):
        algebraic_point = np.asarray(
            [by_solution_name[name] for name in algebraic.variable_names],
            dtype=np.float64,
        )
        algebraic_objective = algebraic.objective.value(algebraic_point)
        if not algebraic.objective.is_linear:
            algebraic_names = set(algebraic.variable_names)
            auxiliary = [
                variable
                for variable in sub_variables
                if variable.name not in algebraic_names and abs(float(variable.getObj())) > 0
            ]
            if len(auxiliary) == 1:
                objective_auxiliary = auxiliary[0].name
    accepted = False
    improved_incumbent = False
    if main_model is not None:
        try:
            infinity = abs(float(main_model.infinity()))
            primal_before = float(main_model.getPrimalbound())
            if not math.isfinite(primal_before) or abs(primal_before) >= 0.99 * infinity:
                primal_before = None
            main_variables = {
                variable.name: variable for variable in main_model.getVars(transformed=False)
            }
            original_before = None
            if algebraic is not None and all(
                name in main_variables for name in algebraic.variable_names
            ):
                incumbent = main_model.getBestSol()
                if incumbent is not None:
                    incumbent_point = np.asarray(
                        [
                            main_model.getSolVal(incumbent, main_variables[name])
                            for name in algebraic.variable_names
                        ],
                        dtype=np.float64,
                    )
                    incumbent_evaluation = algebraic.evaluate(incumbent_point)
                    if incumbent_evaluation.maximum_infeasibility <= 1e-6:
                        original_before = incumbent_evaluation.objective
            if all(variable.name in main_variables for variable in sub_variables):
                # ``sub_variables`` belong to an original-problem copy.  Use
                # an original-space solution here as transformed variables can
                # be locally fixed or aggregated at the current main node.
                translated = main_model.createOrigSol(heuristic)
                for variable, value in zip(sub_variables, solution_values, strict=True):
                    translated_value = float(value)
                    if variable.name == objective_auxiliary and algebraic_objective is not None:
                        translated_value = algebraic_objective
                    main_model.setSolVal(
                        translated,
                        main_variables[variable.name],
                        translated_value,
                    )
                accepted = bool(
                    main_model.trySol(
                        translated,
                        printreason=False,
                        completely=True,
                        checkbounds=True,
                        checkintegrality=True,
                        checklprows=True,
                    )
                )
                if accepted:
                    if algebraic_objective is not None and original_before is not None:
                        tolerance = 1e-9 * max(
                            1.0,
                            abs(original_before),
                            abs(algebraic_objective),
                        )
                        if algebraic.objective_sense == "maximize":
                            improved_incumbent = algebraic_objective > original_before + tolerance
                        else:
                            improved_incumbent = algebraic_objective < original_before - tolerance
                    elif algebraic_objective is not None:
                        improved_incumbent = True
                    else:
                        primal_after = float(main_model.getPrimalbound())
                        if primal_before is None:
                            improved_incumbent = math.isfinite(primal_after)
                        else:
                            tolerance = 1e-9 * max(1.0, abs(primal_before), abs(primal_after))
                            if str(main_model.getObjectiveSense()) == "maximize":
                                improved_incumbent = primal_after > primal_before + tolerance
                            else:
                                improved_incumbent = primal_after < primal_before - tolerance
        except Exception:
            accepted = False
            improved_incumbent = False
    try:
        objective = (
            algebraic_objective
            if algebraic_objective is not None
            else float(sub_model.getSolObjVal(best))
        )
        if not math.isfinite(objective):
            objective = None
    except Exception:
        objective = None
    return CompletionResult(
        True,
        accepted,
        improved_incumbent,
        status,
        objective,
        solution_values,
        perf_counter() - started,
        len(variable_names),
    )


__all__ = [
    "CompletionResult",
    "complete_integer_assignment",
    "create_completion_template",
]
