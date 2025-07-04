from datetime import datetime, date, timedelta
import time
from typing import Optional, Any

from ortools.sat.python import cp_model
from rich.console import Console
from rich.table import Table

from engine.config import config
from engine.models import Activity, TrainingInterval, Worker, Constraints, Constraint, ScheduleResult
from engine.utils import normalize_penalty_list, timer, normalize_penalty

console = Console(color_system="windows")


class SolutionCallback(cp_model.CpSolverSolutionCallback):
    def __init__(self, assignment_vars, activities, worker_ids):
        cp_model.CpSolverSolutionCallback.__init__(self)
        self._assignment_vars = assignment_vars
        self._activities = activities
        self._worker_ids = worker_ids
        self._solution_count = 0
        self._start_time = time.time()

    def on_solution_callback(self):
        self._solution_count += 1
        elapsed = time.time() - self._start_time
        console.print(f"[bold green]Solution {self._solution_count} found after {elapsed:.2f} seconds!")

        total_assigned = 0
        total_expected = 0

        for activity in self._activities:
            if activity.constraintData.expectedNbWorker is not None:
                expected = activity.constraintData.expectedNbWorker
                total_expected += expected

            assigned = sum(
                1
                for worker_id in self._worker_ids
                if (worker_id, activity.id) in self._assignment_vars and self.Value(self._assignment_vars[(worker_id, activity.id)]) == 1
            )
            total_assigned += assigned

        score = total_assigned / total_expected if total_expected > 0 else 1.0
        console.print(f"[bold]Current score: {score:.2%}, Total assigned: {total_assigned}, Total expected: {total_expected}")


def solve_scheduling_problem(activities: list[Activity], workers: list[Worker], constraints: Constraints, debug_mode: bool = False) -> ScheduleResult:
    """
    Main entry point for solving the scheduling problem.

    Args:
        activities: List of activities to schedule
        workers: List of available workers
        constraints: Scheduling constraints and weights
        debug_mode: Whether to enable debug output

    Returns:
        Dictionary with success status, assignments, and statistics
    """
    total_start_time = time.time()
    worker_ids = [w.id for w in workers]  # Ensure worker_ids is a list of IDs
    if debug_mode:
        console.print("[bold blue]Starting optimization model creation...")

    model = cp_model.CpModel()

    # Extract constraint weights
    spread_workers_weight = constraints.spreadWorkers.weight if constraints.spreadWorkers else 0
    workers_availability_weight = constraints.availability.weight if constraints.availability else 0
    worker_levels_weight = constraints.skillLevels.weight if constraints.skillLevels else 0
    pre_assigned_workers_weight = constraints.preAssignedWorkers.weight if constraints.preAssignedWorkers else 0
    localization_weight = constraints.localization.weight if constraints.localization else 0
    trainings_availibility_weight = constraints.trainings.weight if constraints.trainings else 0
    max_shifts_weight = constraints.maxShiftsPerDay.weight if constraints.maxShiftsPerDay else 0

    # Extract constraint data
    worker_workstations_level_constrain = constraints.skillLevels
    pre_assigned_workers_per_activity = constraints.preAssignedWorkers
    localization_constraint = constraints.localization
    worker_availability_constraint = constraints.availability
    max_shifts_constraint = constraints.maxShiftsPerDay
    worker_trainings_availibility_constraint = constraints.trainings

    # Create variables and add constraints
    if debug_mode:
        console.print("[bold green]Creating assignment variables...")

    assignment_vars = create_assignments_variables(model, activities, workers, debug_mode)

    if pre_assigned_workers_weight > 0:
        add_pre_assigned_workers_constraints(model, activities, workers, pre_assigned_workers_per_activity, assignment_vars, debug_mode)

    normalized_localization_penalty = add_localization_constraints_or_penalties(
        model, activities, workers, localization_constraint, assignment_vars, debug_mode
    )
    normalized_availability_penalty = add_availability_constraints_or_penalties(
        model, activities, workers, worker_availability_constraint, assignment_vars, debug_mode
    )
    normalized_availability_penalty_trainings = add_availability_constraints_or_penalties_trainings(
        model, activities, workers, worker_trainings_availibility_constraint, assignment_vars, debug_mode
    )

    add_overlap_constraints(model, activities, worker_ids, assignment_vars)
    add_max_shifts_constraints(model, max_shifts_constraint, assignment_vars, worker_ids, debug_mode)

    normalized_missing_penalty, normalized_spread_penalty, normalized_skill_penalty = add_activity_constraints_and_penalties(
        model,
        activities,
        worker_ids,
        worker_workstations_level_constrain,
        assignment_vars,
        pre_assigned_workers_per_activity,
        int(spread_workers_weight),
        debug_mode,
    )

    model.minimize(
        normalized_missing_penalty * 100
        + normalized_spread_penalty * spread_workers_weight
        + normalized_availability_penalty * workers_availability_weight
        + normalized_localization_penalty * localization_weight
        + normalized_skill_penalty * worker_levels_weight
        + normalized_availability_penalty_trainings * trainings_availibility_weight
    )

    # Solve the model
    if debug_mode:
        console.print("[bold red]Solving optimization model...")
    solve_start_time = time.time()
    solver = cp_model.CpSolver()
    # solver.parameters.log_search_progress = True
    solver.parameters.num_search_workers = 32  # If you're using parallelism

    status = solver.solve(model)
    status_str = solver.status_name(status)

    if debug_mode:
        console.print(f"[bold]Solution status: {status_str}")
        console.print(f"[bold red]Solving took: {time.time() - solve_start_time:.2f} seconds")

    if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
        print("Solution found.")
        print(f"Normalized missing penalty: {solver.value(normalized_missing_penalty * 100)}")
        print(f"Normalized spread penalty: {solver.value(normalized_spread_penalty) * spread_workers_weight}")
        print(f"Normalized skill penalty: {solver.value(normalized_skill_penalty) * worker_levels_weight}")
        print(f"Final penalty : {solver.objective_value}")
        if debug_mode:
            console.print("[bold green]Solver statistics:")
            console.print(f"- Branches: {solver.num_branches}")
            console.print(f"- Conflicts: {solver.num_conflicts}")
            console.print(f"- Wall time: {solver.wall_time:.2f} seconds")
            console.print(f"- Total time: {time.time() - total_start_time:.2f} seconds")

        # Extract assignments
        assignments = {}
        for (worker_id, activity_id), var in assignment_vars.items():
            assignments[(worker_id, activity_id)] = solver.Value(var) == 1

        statistics = {
            "branches": solver.num_branches,
            "conflicts": solver.num_conflicts,
            "wall_time": solver.wall_time,
            "status": status_str,
            "total_time": time.time() - total_start_time,
        }

        return ScheduleResult(
            success=True, assignments=assignments, statistics=statistics, error_message=None, penalty=solver.objective_value, score=0
        )
    else:
        return ScheduleResult(
            success=False,
            assignments={},
            statistics={"status": status_str, "total_time": time.time() - total_start_time},
            error_message=f"No solution found. Status: {status_str}",
            penalty=-1,
            score=0,
        )


@timer
def create_assignments_variables(
    model: cp_model.CpModel,
    activities: list[Activity],
    workers: list[Worker],
    debug_mode: bool = False,
) -> dict[tuple[str, str], cp_model.BoolVarT]:
    assignment_vars: dict[tuple[str, str], cp_model.BoolVarT] = {}
    for activity in activities:
        for worker in workers:
            assignment_vars[(worker.id, activity.id)] = model.new_bool_var(f"assign_{worker.id}_{activity.id}")
    if debug_mode:
        console.print(f"[bold green]Created {len(assignment_vars)} assignment variables")
    return assignment_vars


@timer
def add_pre_assigned_workers_constraints(
    model: cp_model.CpModel,
    activities: list[Activity],
    workers: list[Worker],
    pre_assigned_workers_constraint: Constraint[dict[str, list[str]]],
    assignment_vars: dict[tuple[str, str], cp_model.BoolVarT],
    debug_mode: bool = False,
) -> None:
    if not pre_assigned_workers_constraint.constraintData:
        if debug_mode:
            console.print("[bold green]No pre-assigned workers constraint data, skipping...")
        return
    for activity in activities:
        if activity.id not in pre_assigned_workers_constraint.constraintData:
            continue

        pre_assigned_workers = pre_assigned_workers_constraint.constraintData[activity.id]
        if not pre_assigned_workers:
            continue

        # Ensure that the pre-assigned workers are assigned to the activity
        for worker_id in pre_assigned_workers:
            if worker_id not in [worker.id for worker in workers]:
                continue
            if (worker_id, activity.id) in assignment_vars:
                model.add(assignment_vars[(worker_id, activity.id)] == 1)


@timer
def add_localization_constraints_or_penalties(
    model: cp_model.CpModel,
    activities: list[Activity],
    workers: list[Worker],
    localization_constraint: Constraint[dict[str, dict[str, bool]]],
    assignment_vars: dict[tuple[str, str], cp_model.BoolVarT],
    debug_mode: bool = False,
) -> Optional[cp_model.IntVar]:
    if localization_constraint.weight == 0 or not localization_constraint.constraintData:
        if debug_mode:
            console.print("[bold green]No localization constraints or weight is 0, skipping...")
        return model.new_int_var(0, 0, "zero_localization_penalty")

    permissions_not_allowed = 0
    localization_penalties = []
    max_penalty = 0

    for activity in activities:
        max_penalty += activity.constraintData.expectedNbWorker or 0
        for worker in workers:
            if localization_constraint.weight > 0:
                localization_matrix = localization_constraint.constraintData
                if worker.id not in localization_matrix or activity.workstationId not in localization_matrix[worker.id]:
                    continue

                permission_allowed = localization_matrix[worker.id][activity.workstationId]

                if not permission_allowed:
                    if localization_constraint.weight == 100:
                        # Hard constraint - prevent assignment
                        if (worker.id, activity.id) in assignment_vars:
                            del assignment_vars[(worker.id, activity.id)]
                        # model.add(assignment_vars[(worker.id, activity.id)] == 0)
                        # permissions_not_allowed += 1
                    else:
                        # Soft constraint - add penalty
                        if (worker.id, activity.id) in assignment_vars:
                            penalty = model.new_int_var(0, 1, f"localization_penalty_{worker.id}_{activity.id}")
                            model.add(penalty == assignment_vars[(worker.id, activity.id)])  # penalty = 1 if assignment made
                            localization_penalties.append(penalty)

    if debug_mode:
        console.print(
            f"[bold green]Created {len(localization_penalties)} localization penalties and {permissions_not_allowed} permissions not allowed"
        )

    return (
        normalize_penalty_list(model, localization_penalties, max_penalty, "localization_penalty")
        if localization_penalties
        else model.new_int_var(0, 0, "zero_localization_penalty")
    )


@timer
def add_availability_constraints_or_penalties_trainings(
    model: cp_model.CpModel,
    activities: list[Activity],
    workers: list[Worker],
    worker_availability_trainings_constraint: Constraint[dict[str, list[TrainingInterval]]],
    assignment_vars: dict[tuple[str, str], cp_model.BoolVarT],
    debug_mode: bool = False,
) -> Optional[cp_model.IntVar]:
    if worker_availability_trainings_constraint.weight == 0:
        if debug_mode:
            console.print("[bold green]No worker training availability constraints or weight is 0, skipping...")
        return model.new_int_var(0, 0, "zero_availability_training_penalty")

    availability_trainings_penalties = []
    max_penalty = sum(activity.constraintData.expectedNbWorker or 0 for activity in activities)
    worker_availability_training_matrix = worker_availability_trainings_constraint.constraintData

    for activity in activities:
        # Get start and end dates of the activity
        start_date = datetime.fromisoformat(activity.startDateTime.replace("Z", "+00:00"))
        end_date = datetime.fromisoformat(activity.endDateTime.replace("Z", "+00:00"))

        for worker in workers:
            if worker.id not in worker_availability_training_matrix:
                continue

            training_intervals = worker_availability_training_matrix[worker.id]
            # Use list of intervals to check for overlap with activity
            overlaps = any(
                not (
                    end_date <= datetime.fromisoformat(interval.startDate.replace("Z", "+00:00"))
                    or start_date >= datetime.fromisoformat(interval.endDate.replace("Z", "+00:00"))
                )
                for interval in training_intervals
            )

            if overlaps:
                if worker_availability_trainings_constraint.weight == 100:
                    if (worker.id, activity.id) in assignment_vars:
                        del assignment_vars[(worker.id, activity.id)]
                    # model.add(assignment_vars[(worker.id, activity.id)] == 0)
                    # continue
                elif worker_availability_trainings_constraint.weight > 0:
                    if (worker.id, activity.id) in assignment_vars:
                        penalty = model.new_int_var(0, 1, f"availability_penalty_training_{worker.id}_{activity.id}")
                        model.add(penalty == assignment_vars[(worker.id, activity.id)])
                        availability_trainings_penalties.append(penalty)

    if debug_mode:
        console.print(f"[bold green]Created {len(availability_trainings_penalties)} availability training penalties")

    return (
        normalize_penalty_list(model, availability_trainings_penalties, max_penalty, "availibility_trainings_penalty")
        if availability_trainings_penalties
        else model.new_int_var(0, 0, "zero_availibility_training_penalty")
    )


@timer
def add_availability_constraints_or_penalties(
    model: cp_model.CpModel,
    activities: list[Activity],
    workers: list[Worker],
    worker_availability_constraint: Constraint[dict[str, list[str]]],
    assignment_vars: dict[tuple[str, str], cp_model.BoolVarT],
    debug_mode: bool = False,
) -> Optional[cp_model.IntVar]:
    if worker_availability_constraint.weight == 0:
        if debug_mode:
            console.print("[bold green]No worker availability constraints or weight is 0, skipping...")
        return model.new_int_var(0, 0, "zero_availability_penalty")

    availability_penalties = []
    max_penalty = sum(activity.constraintData.expectedNbWorker or 0 for activity in activities)
    worker_availability_matrix = worker_availability_constraint.constraintData

    for activity in activities:
        # Get start and end dates of the activity
        start_date = datetime.fromisoformat(activity.startDateTime.replace("Z", "+00:00")).date()
        end_date = datetime.fromisoformat(activity.endDateTime.replace("Z", "+00:00")).date()

        # Create a list of all days the activity spans
        activity_days = [start_date + timedelta(days=x) for x in range((end_date - start_date).days + 1)]

        for worker in workers:
            if worker.id not in worker_availability_matrix:
                continue

            worker_days = worker_availability_matrix[worker.id]
            # Check if worker is available on ANY of the activity days
            if not any(date.fromisoformat(day) in activity_days for day in worker_days):
                continue

            if worker_availability_constraint.weight == 100:
                if (worker.id, activity.id) in assignment_vars:
                    del assignment_vars[(worker.id, activity.id)]
                # model.add(assignment_vars[(worker.id, activity.id)] == 0)
                continue
            elif worker_availability_constraint.weight > 0:
                if (worker.id, activity.id) in assignment_vars:
                    penalty = model.new_int_var(0, 1, f"availability_penalty_{worker.id}_{activity.id}")
                    model.add(penalty == assignment_vars[(worker.id, activity.id)])
                    availability_penalties.append(penalty)

    if debug_mode:
        console.print(f"[bold green]Created {len(availability_penalties)} availability penalties")

    return (
        normalize_penalty_list(model, availability_penalties, max_penalty, "availibility_penalty")
        if availability_penalties
        else model.new_int_var(0, 0, "zero_availibility_penalty")
    )


@timer
def add_max_shifts_constraints(
    model,
    max_shifts_constraint,
    assignment_vars,
    worker_ids,
    debug_mode: bool = False,
):
    if max_shifts_constraint.weight == 0:
        if debug_mode:
            console.print("[bold green]No max shifts constraints or weight is 0, skipping...")
    elif max_shifts_constraint.weight == 100:
        worker_activities_matrix = max_shifts_constraint.constraintData.workerActivitiesMatrix
        shifts_ids = max_shifts_constraint.constraintData.shiftIds
        nb_days = len(max_shifts_constraint.constraintData.scheduleDays)
        shift_transition_matrix = max_shifts_constraint.constraintData.shiftTransitionMatrix
        shifts_per_day = len(shifts_ids)

        for worker_id in worker_ids:
            for day_idx in range(nb_days):
                # For this worker & day, get assignment vars grouped by shift
                shift_activity_vars = []

                for shift_idx in range(shifts_per_day):
                    slot_idx = day_idx * shifts_per_day + shift_idx
                    activities_in_shift = worker_activities_matrix[worker_id][slot_idx]  # list of {activityId, shiftId}

                    # Get assignment vars for this worker for these activities
                    vars_for_shift = [
                        assignment_vars[(worker_id, activity["activityId"])]
                        for activity in activities_in_shift
                        if (worker_id, activity["activityId"]) in assignment_vars
                    ]

                    # If no activities assigned for this shift, skip
                    if not vars_for_shift:
                        continue

                    # Create a bool var representing whether worker works this shift (any activity assigned)
                    shift_assigned_var = model.new_bool_var(f"{worker_id}_day{day_idx}_shift{shift_idx}")

                    model.add_max_equality(shift_assigned_var, vars_for_shift)

                    shift_activity_vars.append(shift_assigned_var)

                if shift_activity_vars:
                    model.add(sum(shift_activity_vars) <= 1)
            # add 11 hours between shifts constraint
            for day in range(nb_days - 1):  # Last day has no "next day"
                for shift_from_idx, shift_from_id in enumerate(shifts_ids):
                    from_slot = day * shifts_per_day + shift_from_idx
                    from_activities = worker_activities_matrix[worker_id][from_slot]
                    if not from_activities:
                        continue
                    for shift_to_idx, shift_to_id in enumerate(shifts_ids):
                        to_slot = (day + 1) * shifts_per_day + shift_to_idx
                        to_activities = worker_activities_matrix[worker_id][to_slot]
                        if not to_activities:
                            continue
                        # Skip valid transitions (true in matrix) and same shift
                        if shift_from_id == shift_to_id or shift_transition_matrix[shift_from_id][shift_to_id]:
                            continue
                        # Build OR for assignments on those shifts
                        from_vars = [
                            assignment_vars[(worker_id, act["activityId"])]
                            for act in from_activities
                            if (worker_id, act["activityId"]) in assignment_vars
                        ]
                        to_vars = [
                            assignment_vars[(worker_id, act["activityId"])]
                            for act in to_activities
                            if (worker_id, act["activityId"]) in assignment_vars
                        ]
                        if not from_vars or not to_vars:
                            continue
                        # Worker cannot be assigned to both shift_from and shift_to
                        from_assigned = model.new_bool_var(f"{worker_id}_day{day}_{shift_from_id}")
                        to_assigned = model.new_bool_var(f"{worker_id}_day{day + 1}_{shift_to_id}")
                        model.add_bool_or(from_vars).only_enforce_if(from_assigned)
                        model.add_bool_and([v.Not() for v in from_vars]).only_enforce_if(from_assigned.Not())
                        model.add_bool_or(to_vars).only_enforce_if(to_assigned)
                        model.add_bool_and([v.Not() for v in to_vars]).only_enforce_if(to_assigned.Not())
                        # Add hard constraint: both can't be true
                        model.add_bool_or([from_assigned.Not(), to_assigned.Not()])

    else:
        raise ValueError("Invalid weight for max shifts constraint")


@timer
def add_overlap_constraints(
    model: cp_model.CpModel,
    activities: list[Activity],
    worker_ids: list[str],
    assignment_vars: dict,
    debug_mode: bool = False,
):
    with console.status("[bold blue]Adding overlap constraints..."):
        overlap_count = 0

        # Pre-compute overlapping activity pairs to avoid repeated checks
        overlapping_pairs = []
        for i in range(len(activities)):
            for j in range(i + 1, len(activities)):
                activity1, activity2 = activities[i], activities[j]
                if activity1.startDateTime < activity2.endDateTime and activity2.startDateTime < activity1.endDateTime:
                    overlapping_pairs.append((activity1, activity2))

        # Add constraints only for overlapping activities
        for worker_id in worker_ids:
            for activity1, activity2 in overlapping_pairs:
                if (worker_id, activity1.id) in assignment_vars and (worker_id, activity2.id) in assignment_vars:
                    model.add(assignment_vars[(worker_id, activity1.id)] + assignment_vars[(worker_id, activity2.id)] <= 1)
                    overlap_count += 1

        if debug_mode:
            console.print(f"[bold blue]Added {overlap_count} overlap constraints")


@timer
def add_missing_worker_penalties(
    model: cp_model.CpModel,
    activities: list[Activity],
    worker_ids: list[str],
    assignment_vars: dict[tuple[str, str], cp_model.BoolVarT],
    pre_assigned_workers_per_activity: dict[str, list[str]],
    debug_mode: bool = False,
) -> cp_model.IntVar:
    """Softly penalize under-staffing"""

    # ---------------- helper -------------------------------------------------
    def load_expr(act_id: str) -> cp_model.LinearExpr:
        return cp_model.LinearExpr.sum([assignment_vars[(w, act_id)] for w in worker_ids if (w, act_id) in assignment_vars])

    total_shortage_terms: list[cp_model.IntVar] = []
    total_expected = 0

    for act in activities:
        exp = act.constraintData.expectedNbWorker or 0

        # # ---------- activities that expect nobody ----------------------------
        # if exp == 0:
        #     if pre_assigned_workers_per_activity:
        #         # honor any locked-in workers
        #         model.add(load_expr(act.id) == len(pre_assigned_workers_per_activity.get(act.id, [])))
        #     continue

        total_expected += exp

        load = load_expr(act.id)

        model.add(load <= exp)  # forbid over-staffing

        shortage = model.new_int_var(0, exp, f"shortage_{act.id}")
        model.add(shortage == exp - load)

        total_shortage_terms.append(shortage)

    total_shortage = model.new_int_var(0, total_expected, "total_shortage")
    model.add(total_shortage == cp_model.LinearExpr.Sum(total_shortage_terms))

    return normalize_penalty(
        model,
        total_shortage,
        total_expected,
        "normalized_missing_penalty",
    )


@timer
def add_skill_level_penalties(
    model: cp_model.CpModel,
    activities: list[Activity],
    worker_ids: list[str],
    worker_workstations_level_constraint: Constraint,
    assignment_vars: dict[tuple[str, str], cp_model.BoolVarT],
    debug_mode: bool = False,
) -> cp_model.IntVar:
    """Add penalties for skill level requirements."""
    if worker_workstations_level_constraint.weight == 0:
        if debug_mode:
            console.print("[bold green]No skill level constraints or weight is 0, skipping...")
        zero = model.new_int_var(0, 0, "zero_skill_penalty")
        return zero
    worker_workstations_level_matrix = worker_workstations_level_constraint.constraintData
    total_levels_required = 0
    skill_penalties = []

    for activity in activities:
        if not activity.constraintData.skillLevelRequirements:
            continue

        for level_requirement in activity.constraintData.skillLevelRequirements:
            level = f"LEVEL{level_requirement.skillLevel}"
            total_levels_required += level_requirement.minWorkers

            # Find eligible workers for this level
            eligible_worker_vars = []
            non_eligible_worker_vars = []

            for worker_id in worker_ids:
                if worker_workstations_level_matrix and worker_workstations_level_matrix[worker_id][activity.workstationId] == level:
                    if (worker_id, activity.id) in assignment_vars:
                        eligible_worker_vars.append(assignment_vars[(worker_id, activity.id)])
                else:
                    if (worker_id, activity.id) in assignment_vars:
                        non_eligible_worker_vars.append(assignment_vars[(worker_id, activity.id)])

            # Handle hard constraint (weight = 100)
            if worker_workstations_level_constraint.weight == 100:
                model.add(sum(eligible_worker_vars) >= level_requirement.minWorkers)
                for non_eligible in non_eligible_worker_vars:
                    # Instead of deleting, just skip using them
                    pass
            else:
                # Track number of qualified workers assigned
                assigned_var = model.new_int_var(
                    0,
                    len(worker_ids),
                    f"assigned_{activity.id}_level{level_requirement.skillLevel}",
                )
                model.add(assigned_var == sum(eligible_worker_vars))

                # Calculate shortage of required workers
                shortage = model.new_int_var(
                    0,
                    level_requirement.minWorkers,
                    f"shortage_{activity.id}_level{level_requirement.skillLevel}",
                )
                model.add(shortage == level_requirement.minWorkers - assigned_var)
                skill_penalties.append(shortage)

    if total_levels_required > 0 and skill_penalties:
        return normalize_penalty_list(
            model,
            skill_penalties,
            total_levels_required * len(activities),
            "normalized_skill_penalty",
        )

    return model.new_int_var(0, 0, "zero_skill_penalty")


@timer
def add_workload_spread_penalties(
    model: cp_model.CpModel,
    activities: list[Activity],
    worker_ids: list[str],
    assignment_vars: dict[tuple[str, str], cp_model.BoolVarT],
    spread_workers_weight: int,
    debug_mode: bool = False,
) -> cp_model.IntVar:
    """
    Add penalties for uneven workload distribution.
    Returns either imbalance or a normalized penalty variable. Depending on the config
    """

    total_expected_workers = sum(activity.constraintData.expectedNbWorker or 0 for activity in activities)

    # Soft constraints only
    if spread_workers_weight == 100:
        imbalance = 0
        if debug_mode:
            console.print("using imbalance method, it might have some unexpected issues")
        min_assignments = model.new_int_var(0, len(activities), "min_worker_load")
        max_assignments = model.new_int_var(0, len(activities), "max_worker_load")
        worker_loads = []

        # Calculate the load for each worker
        for worker_id in worker_ids:
            assignments = sum(assignment_vars[(worker_id, act.id)] for act in activities if (worker_id, act.id) in assignment_vars)
            worker_load = model.new_int_var(0, len(activities), f"worker_load_{worker_id}")
            model.add(worker_load == assignments)
            worker_loads.append(worker_load)

        # Find min and max assignments
        model.AddMinEquality(min_assignments, worker_loads)
        model.add_max_equality(max_assignments, worker_loads)

        # Enforce maximum difference between min and max assignments
        max_allowed_difference = 1  # Workers having a difference of 1 assignment is acceptable (more should not happen)
        model.add(max_assignments - min_assignments <= max_allowed_difference)
        return model.new_int_var(0, 0, "zero_spread_penalty")

    else:
        penalties: list[cp_model.IntVar] = []
        weights = [round(x**1.5) for x in range(len(activities) + 1)]  # f has to be such as f(x1) + f(x2) < f(x1 + x2)
        worker_loads = []
        for worker_id in worker_ids:
            assignments = sum(assignment_vars[(worker_id, act.id)] for act in activities if (worker_id, act.id) in assignment_vars)
            worker_load = model.new_int_var(0, len(activities), f"worker_load_{worker_id}")
            model.add(worker_load == assignments)
            worker_loads.append(worker_load)
            penalty = model.new_int_var(0, max(weights), f"worker_penalty_{worker_id}")
            model.add_element(worker_load, weights, penalty)
            penalties.append(penalty)

        max_expected_for_an_activity = max(
            (activity.constraintData.expectedNbWorker or 0 for activity in activities), default=0
        )  # 0 if the list is empty
        return normalize_penalty_list(
            model, penalties, max_expected_for_an_activity * max(weights) + 1, "normalized_spread_penalty"
        )  # the upper bound could be reduced


@timer
def add_activity_constraints_and_penalties(
    model: cp_model.CpModel,
    activities: list[Activity],
    worker_ids: list[str],
    worker_workstations_level_constraint: Constraint,
    assignment_vars: dict[tuple[str, str], cp_model.BoolVarT],
    pre_assigned_workers_constraint: Constraint,
    spread_workers_weight: int,
    debug_mode: bool = False,
):
    with console.status("[bold yellow]Adding activity constraints..."):
        if debug_mode:
            table = Table(title="Activity Constraints")
            table.add_column("Activity ID", style="cyan")
            table.add_column("Expected Workers", style="green")
            table.add_column("Skill Requirements", style="yellow")
            table.add_column("Eligible Workers", style="blue")
            # Table population would go here if needed

        normalized_missing_penalty = add_missing_worker_penalties(
            model,
            activities,
            worker_ids,
            assignment_vars,
            pre_assigned_workers_constraint.constraintData if pre_assigned_workers_constraint else {},
            debug_mode,
        )

        normalized_skill_penalty = add_skill_level_penalties(
            model,
            activities,
            worker_ids,
            worker_workstations_level_constraint,
            assignment_vars,
            debug_mode,
        )

        normalized_spread_penalty = add_workload_spread_penalties(
            model,
            activities,
            worker_ids,
            assignment_vars,
            spread_workers_weight,
            debug_mode,
        )

        if debug_mode:
            console.print(table)

        return (
            normalized_missing_penalty,
            normalized_spread_penalty,
            normalized_skill_penalty,
        )
