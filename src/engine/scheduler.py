from engine.engine import solve_scheduling_problem
from engine.models import Activity, Worker, Constraints, FilledActivity, SchedulingResultWithFilledActivities
from rich.progress import Progress, BarColumn, TextColumn


def _create_filled_activities(activities: list[Activity], worker_ids: list[str], assignments: dict[tuple[str, str], bool]) -> list[FilledActivity]:
    """
    Convert the assignment results into filled activities

    Args:
        activities: Original activities
        worker_ids: List of worker IDs
        assignments: Dictionary mapping (worker_id, activity_id) -> bool

    Returns:
        List of FilledActivity objects
    """
    filled_activities = []

    for activity in activities:
        worker_assignments = []

        for worker_id in worker_ids:
            if assignments.get((worker_id, activity.id), False):
                worker_assignments.append(worker_id)

        filled_activity = FilledActivity(
            id=activity.id,
            workstationId=activity.workstationId,
            constraintData=activity.constraintData,
            startDateTime=activity.startDateTime,
            endDateTime=activity.endDateTime,
            assignedWorkerIds=worker_assignments,
        )
        filled_activities.append(filled_activity)

    return filled_activities


def _print_input_summary(activities: list[Activity], workers: list[Worker], constraints: Constraints, debug_mode: bool = False):
    """Print a summary of the input data"""
    print("=" * 50)
    print("AUTO-SCHEDULE OPTIMIZATION STARTING")
    print("=" * 50)
    print(f"Activities: {len(activities)}")
    print(f"Workers: {len(workers)}")

    if constraints.skillLevels and constraints.skillLevels.constraintData:
        workstations: set[str] = set()
        for worker_data in constraints.skillLevels.constraintData.values():
            workstations.update(worker_data.keys())
        print(f"Workstations: {len(workstations)}")
    else:
        print("Workstations: 0")
    print("-" * 50)

    if debug_mode:
        # Print constraints and their weights with a visual slider representation
        print("Constraints and Weights:\n")
        with Progress(
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TextColumn("[bold blue]{task.percentage:>3.0f}%"),
        ) as progress:
            if constraints.spreadWorkers:
                progress.add_task("Spread Workers", total=100, completed=int(constraints.spreadWorkers.weight))
            if constraints.availability:
                progress.add_task("Worker Availability", total=100, completed=int(constraints.availability.weight))
            if constraints.skillLevels:
                progress.add_task("Skill Levels", total=100, completed=int(constraints.skillLevels.weight))
            if constraints.preAssignedWorkers:
                progress.add_task("Pre-assigned Workers", total=100, completed=int(constraints.preAssignedWorkers.weight))
            if constraints.localization:
                progress.add_task("Localization", total=100, completed=int(constraints.localization.weight))
            if constraints.maxShiftsPerDay:
                progress.add_task("Max Shifts Per Day", total=100, completed=int(constraints.maxShiftsPerDay.weight))
            if constraints.trainings:
                progress.add_task("Workers Trainings", total=100, completed=int(constraints.trainings.weight))
        print("-" * 80)


class SchedulingEngine:
    def __init__(self, debug: bool = False):
        self.debug_mode = debug

    def schedule(self, activities: list[Activity], workers: list[Worker], constraints: Constraints) -> SchedulingResultWithFilledActivities:
        """
        Main scheduling method that coordinates the entire optimization process
        """
        if not constraints:
            return SchedulingResultWithFilledActivities(
                filled_activities=[], score=0.0, success=False, error_message="No constraints provided", assignments={}, statistics={}
            )
        worker_ids = [w.id for w in workers]
        _print_input_summary(activities, workers, constraints, debug_mode=self.debug_mode)

        # Call the core engine
        result = solve_scheduling_problem(activities=activities, workers=workers, constraints=constraints, debug_mode=self.debug_mode)

        if result.success:
            filled_activities = _create_filled_activities(activities=activities, worker_ids=worker_ids, assignments=result.assignments)

            # Calculate final score
            total_expected = sum(a.constraintData.expectedNbWorker for a in activities if a.constraintData.expectedNbWorker is not None)
            total_assigned = sum(len(fa.assignedWorkerIds) for fa in filled_activities)
            score = (total_assigned / total_expected) if total_expected > 0 else 1.0

            print(f"[FINAL] Score: {score:.2%}, Assigned: {total_assigned}, Expected: {total_expected}")

            return SchedulingResultWithFilledActivities(
                filled_activities=filled_activities,
                score=score,
                success=True,
                error_message=None,
                assignments=result.assignments,
                statistics=result.statistics,
                penalty=result.penalty,
            )
        else:
            print("[ERROR] No solution found!")
            return SchedulingResultWithFilledActivities(
                filled_activities=[], score=0.0, success=False, error_message=result.error_message, assignments={}, statistics={}, penalty=result.penalty
            )
