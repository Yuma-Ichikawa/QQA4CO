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
    minimum_relative_improvement: float = 0.0,
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
    if (
        not math.isfinite(minimum_relative_improvement)
        or not 0 <= minimum_relative_improvement < 1
    ):
        raise ValueError("minimum_relative_improvement must be in [0, 1).")
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
    primal_before = None
    if main_model is not None:
        with suppress(Exception):
            infinity = abs(float(main_model.infinity()))
            raw_primal_before = float(main_model.getPrimalbound())
            if math.isfinite(raw_primal_before) and abs(raw_primal_before) < 0.99 * infinity:
                primal_before = raw_primal_before
                tolerance = max(1e-9, minimum_relative_improvement) * max(
                    1.0, abs(primal_before)
                )
                objective_limit = (
                    primal_before + tolerance
                    if str(main_model.getObjectiveSense()) == "maximize"
                    else primal_before - tolerance
                )
                sub_model.setObjlimit(objective_limit)
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
    try:
        candidate_objective = (
            algebraic_objective
            if algebraic_objective is not None
            else float(sub_model.getSolObjVal(best))
        )
        if not math.isfinite(candidate_objective):
            candidate_objective = None
    except Exception:
        candidate_objective = None
    if primal_before is not None and candidate_objective is not None:
        tolerance = max(1e-9, minimum_relative_improvement) * max(
            1.0, abs(primal_before)
        )
        improves = (
            candidate_objective > primal_before + tolerance
            if str(main_model.getObjectiveSense()) == "maximize"
            else candidate_objective < primal_before - tolerance
        )
        if not improves:
            return CompletionResult(
                True,
                False,
                False,
                "nonimproving",
                candidate_objective,
                solution_values,
                perf_counter() - started,
                len(variable_names),
            )
    accepted = False
    improved_incumbent = False
    if main_model is not None:
        try:
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
                        printreason=verbose,
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
    return CompletionResult(
        True,
        accepted,
        improved_incumbent,
        status,
        candidate_objective,
        solution_values,
        perf_counter() - started,
        len(variable_names),
    )


def complete_integer_assignment_dive(
    model,
    variables: Sequence,
    values: Sequence[float],
    *,
    heuristic=None,
    algebraic: AlgebraicModel | None = None,
    lp_iterations: int = 500,
    anchor_values: Sequence[float] | None = None,
    change_order: Sequence[int] | None = None,
    max_repair_changes: int = 12,
    minimum_relative_improvement: float = 0.0,
) -> CompletionResult:
    """Complete an integer assignment with the active node LP in-place.

    SCIP diving temporarily fixes transformed integer variables, reoptimises
    the already loaded node LP, and restores the original node afterwards.
    This avoids constructing a complete sub-SCIP for every QQA candidate.  It
    is an exact continuous completion for linear MIPs; ``trySol`` remains the
    final authority for all original constraints and integrality conditions.
    """
    if len(variables) != len(values):
        raise ValueError("variables and values must have the same length.")
    if len({variable.name for variable in variables}) != len(variables):
        raise ValueError("variables must be unique.")
    if isinstance(lp_iterations, bool) or not isinstance(lp_iterations, int) or lp_iterations < 1:
        raise ValueError("lp_iterations must be a positive integer.")
    if (
        isinstance(max_repair_changes, bool)
        or not isinstance(max_repair_changes, int)
        or max_repair_changes < 0
    ):
        raise ValueError("max_repair_changes must be a non-negative integer.")
    if (
        not math.isfinite(minimum_relative_improvement)
        or not 0 <= minimum_relative_improvement < 1
    ):
        raise ValueError("minimum_relative_improvement must be in [0, 1).")
    started = perf_counter()
    candidate = None
    unsuccessful_status = "lp_cutoff_or_infeasible"
    incumbent = model.getBestSol()
    fixed_values = np.asarray(values, dtype=np.float64)
    anchor = None if anchor_values is None else np.rint(np.asarray(anchor_values, dtype=np.float64))
    if anchor is not None and anchor.shape != fixed_values.shape:
        raise ValueError("anchor_values must have the same shape as values.")
    if change_order is None:
        ordered_changes = list(range(len(variables)))
    else:
        ordered_changes = [int(index) for index in change_order]
        if len(set(ordered_changes)) != len(ordered_changes) or any(
            index < 0 or index >= len(variables) for index in ordered_changes
        ):
            raise ValueError("change_order must contain unique valid positions.")
    primal_before = None
    try:
        model.startDive()
        try:
            local_lower = np.asarray(
                [float(model.getVarLbDive(variable)) for variable in variables]
            )
            local_upper = np.asarray(
                [float(model.getVarUbDive(variable)) for variable in variables]
            )
            if np.any(fixed_values < local_lower - 1e-7) or np.any(
                fixed_values > local_upper + 1e-7
            ):
                return CompletionResult(
                    False,
                    False,
                    False,
                    "bound_infeasible",
                    None,
                    None,
                    perf_counter() - started,
                    len(variables),
                )
            if anchor is not None and (
                np.any(anchor < local_lower - 1e-7) or np.any(anchor > local_upper + 1e-7)
            ):
                anchor = None

            def fix(position: int, value: float) -> None:
                variable = variables[position]
                fixed = float(value)
                current_lower = float(model.getVarLbDive(variable))
                if fixed < current_lower:
                    model.chgVarLbDive(variable, fixed)
                    model.chgVarUbDive(variable, fixed)
                else:
                    model.chgVarUbDive(variable, fixed)
                    model.chgVarLbDive(variable, fixed)

            working = fixed_values.copy() if anchor is None else anchor.copy()
            for position, value in enumerate(working):
                fix(position, value)
            changed = [
                position
                for position in ordered_changes
                if anchor is not None and abs(fixed_values[position] - anchor[position]) > 0.5
            ][:max_repair_changes]
            solve_count = 2 + len(changed)
            iteration_limit = max(20, lp_iterations // max(1, solve_count))
            if anchor is None:
                lp_error, cutoff = model.solveDiveLP(itlim=iteration_limit)
            elif changed:
                for position in changed:
                    fix(position, fixed_values[position])
                lp_error, cutoff = model.solveDiveLP(itlim=iteration_limit)
                if not lp_error and not cutoff:
                    working[changed] = fixed_values[changed]
                elif cutoff and not lp_error:
                    for position in changed:
                        fix(position, anchor[position])
                    accepted_change = False
                    for position in changed:
                        fix(position, fixed_values[position])
                        trial_error, trial_cutoff = model.solveDiveLP(itlim=iteration_limit)
                        if trial_error:
                            lp_error = True
                            break
                        if trial_cutoff:
                            fix(position, anchor[position])
                        else:
                            working[position] = fixed_values[position]
                            accepted_change = True
                    if not lp_error and accepted_change:
                        lp_error, cutoff = model.solveDiveLP(itlim=iteration_limit)
                    else:
                        cutoff = True
            else:
                lp_error, cutoff = False, True
            if not lp_error and not cutoff:
                candidate = model.createSol(heuristic, initlp=True)
                for variable in model.getVars(transformed=True):
                    if variable.isInLP():
                        value = float(model.getSolVal(None, variable))
                    elif incumbent is not None:
                        value = float(model.getSolVal(incumbent, variable))
                    else:
                        lower = float(variable.getLbLocal())
                        upper = float(variable.getUbLocal())
                        value = min(max(0.0, lower), upper)
                    model.setSolVal(candidate, variable, value)
                for variable, value in zip(variables, working, strict=True):
                    model.setSolVal(candidate, variable, float(value))
            elif anchor is not None and not changed:
                unsuccessful_status = "no_integer_change"
            else:
                lp_status = int(model.getLPSolstat())
                unsuccessful_status = {
                    2: "lp_infeasible",
                    4: "objective_cutoff",
                    5: "lp_iteration_limit",
                    6: "lp_time_limit",
                    7: "lp_error",
                }.get(lp_status, "lp_cutoff_or_infeasible")
        finally:
            model.endDive()
    except Exception:
        return CompletionResult(
            False,
            False,
            False,
            "dive_failed",
            None,
            None,
            perf_counter() - started,
            len(variables),
        )
    if candidate is None:
        return CompletionResult(
            False,
            False,
            False,
            unsuccessful_status,
            None,
            None,
            perf_counter() - started,
            len(variables),
        )

    objective = None
    solution_values = None
    original_variables = tuple(model.getVars(transformed=False))
    if algebraic is not None:
        by_name = {variable.name: variable for variable in original_variables}
        if all(name in by_name for name in algebraic.variable_names):
            solution_values = np.asarray(
                [model.getSolVal(candidate, by_name[name]) for name in algebraic.variable_names],
                dtype=np.float64,
            )
            evaluation = algebraic.evaluate(solution_values)
            if evaluation.maximum_infeasibility <= 1e-6:
                objective = float(evaluation.objective)
    try:
        infinity = abs(float(model.infinity()))
        raw_primal_before = float(model.getPrimalbound())
        if math.isfinite(raw_primal_before) and abs(raw_primal_before) < 0.99 * infinity:
            primal_before = raw_primal_before
        feasible = bool(
            model.checkSol(
                candidate,
                printreason=False,
                completely=True,
                checkbounds=True,
                checkintegrality=True,
                checklprows=True,
            )
        )
        cutoff_objective = objective
        if cutoff_objective is None:
            with suppress(Exception):
                cutoff_objective = float(model.getSolObjVal(candidate))
        if feasible and primal_before is not None and cutoff_objective is not None:
            tolerance = max(1e-9, minimum_relative_improvement) * max(
                1.0,
                abs(primal_before),
            )
            improves_enough = (
                cutoff_objective > primal_before + tolerance
                if str(model.getObjectiveSense()) == "maximize"
                else cutoff_objective < primal_before - tolerance
            )
            if not improves_enough:
                return CompletionResult(
                    True,
                    False,
                    False,
                    "nonimproving",
                    objective,
                    solution_values,
                    perf_counter() - started,
                    len(variables),
                )
        accepted = bool(
            feasible
            and model.trySol(
                candidate,
                printreason=False,
                completely=False,
                checkbounds=False,
                checkintegrality=False,
                checklprows=False,
            )
        )
        primal_after = float(model.getPrimalbound()) if accepted else primal_before
    except Exception:
        feasible = False
        accepted = False
        primal_after = primal_before
    improved = accepted and primal_before is None
    if accepted and primal_before is not None and primal_after is not None:
        tolerance = 1e-9 * max(1.0, abs(primal_after), abs(primal_before))
        improved = (
            primal_after > primal_before + tolerance
            if str(model.getObjectiveSense()) == "maximize"
            else primal_after < primal_before - tolerance
        )
    return CompletionResult(
        feasible,
        accepted,
        improved,
        "accepted" if accepted else "feasible" if feasible else "rejected",
        objective,
        solution_values,
        perf_counter() - started,
        len(variables),
    )


__all__ = [
    "CompletionResult",
    "complete_integer_assignment",
    "complete_integer_assignment_dive",
    "create_completion_template",
]
