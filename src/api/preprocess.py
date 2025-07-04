from rich.console import Console
from rich.panel import Panel

from api.models import ScheduleRequest

# preAssignedWorkersPerActivity = dict[str, list[str]]  # activityId → workerIds
# WorkerWorkstationMap = dict[str, WorkerLevels]  # workstationId → level → worker IDs
# WorkerLevels = dict[str, list[str]]  # level → worker IDs
# class ScheduleRequest(BaseModel):
#     activities: list[Activity]
#     workers: list[Worker]
#     workerWorkstationsLevelMatrix: WorkerWorkstationMap
#     assignedWorkersPerActivity: preAssignedWorkersPerActivity

console = Console()

MAXASSIGNMENT = 1000
MAXLEVELS = 4


def preprocess_checks(request: ScheduleRequest):
    """Perform preprocessing checks on the request data."""
    console.print("[bold blue]Running preprocessing checks...[/bold blue]")
    warnings = []
    critical_issues = []

    # Check for empty activities or workers
    if not request.activities:
        raise ValueError("No activities provided.")
    if not request.workers:
        raise ValueError("No workers provided.")

    # Check for activities with missing data
    for activity in request.activities:
        if not activity.id:
            critical_issues.append(f"Activity missing ID: {activity}")
        if not activity.workstationId:
            critical_issues.append(f"Activity {activity.id} missing workstation ID")
        if not hasattr(activity, "constraintData") or not activity.constraintData:
            warnings.append(f"Activity {activity.id} missing constraint data")
        elif activity.constraintData.expectedNbWorker is not None and not 0 <= activity.constraintData.expectedNbWorker < MAXASSIGNMENT:
            warnings.append(f"Activity {activity.id} has invalid expected worker count: {activity.constraintData.expectedNbWorker}")

    # Check for workers with missing data
    for worker in request.workers:
        if not worker.id:
            critical_issues.append(f"Worker missing ID: {worker}")

    all_workstation_ids = [a.workstationId for a in request.activities]
    console.print(f"Found {len(all_workstation_ids)} workstations in activities")

    # # Check worker levels on workstations
    # for workstation_id, worker_per_level in request.workerWorkstationsLevelMatrix.items():
    #     # check that workstation exists
    #     if workstation_id not in all_workstation_ids:
    #         critical_issues.append(f"Workstation {workstation_id} not found in workstations list")
    #         continue
    #     # check that all levels are between 0 and 4 and workers exist
    #     for level, worker_ids in worker_per_level.items():
    #         if level not in [f"LEVEL_{i}" for i in range(MAXLEVELS + 1)]:
    #             critical_issues.append(f"Invalid level {level} for workstation {workstation_id}")
    #         for worker_id in worker_ids:
    #             if not any(worker.id == worker_id for worker in request.workers):
    #                 critical_issues.append(
    #                     f"Worker {worker_id} not found in workers list for workstation {workstation_id}")

    # Check pre assignments
    preAssignedWorkersPerActivity = request.options.constraints.preAssignedWorkers

    for activity_id, worker_ids in preAssignedWorkersPerActivity.constraintData.items():
        # check if activity exists
        if not any(activity.id == activity_id for activity in request.activities):
            critical_issues.append(f"Activity {activity_id} not found in activities list")
            continue
        # check if all workers exist
        for worker_id in worker_ids:
            if not any(worker.id == worker_id for worker in request.workers):
                critical_issues.append(f"Worker {worker_id} not found in workers list for activity {activity_id}")

    # Check for workstation assignments

    # Display warnings and critical issues
    if warnings:
        console.print(
            Panel(
                "\n".join([f"⚠️  {w}" for w in warnings]),
                title="[yellow]Preprocessing Warnings[/yellow]",
                border_style="yellow",
            )
        )

    if critical_issues:
        console.print(
            Panel(
                "\n".join([f"❌ {issue}" for issue in critical_issues]),
                title="[bold red]Critical Issues[/bold red]",
                border_style="red",
            )
        )
        raise ValueError(f"Found {len(critical_issues)} critical issues in input data")

    console.print(f"[green]Preprocessing completed with {len(warnings)} warnings[/green]")
