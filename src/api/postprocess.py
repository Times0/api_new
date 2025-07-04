import os

from ortools.sat.python import cp_model
from rich.console import Console
from rich.table import Table

from engine.models import Activity, FilledActivity
from api.visualize_results import (
    visualize_activity_assignments,
    visualize_worker_distribution,
)

console = Console()


def create_filled_activities(
    activities: list[Activity],
    worker_ids: list[str],
    assignment_vars: dict,
    solver: cp_model.CpSolver,
) -> list[FilledActivity]:
    filled_activities = []

    with console.status("[bold purple]Creating filled activities..."):
        table = Table(title="Worker Assignments")
        table.add_column("Activity ID", style="cyan")
        table.add_column("Expected Workers", style="green")
        table.add_column("Assigned Workers", style="yellow")
        table.add_column("Fill Rate", style="red")

        for activity in activities:
            filled = FilledActivity(
                id=activity.id,
                workstationId=activity.workstationId,
                constraintData=activity.constraintData,
                startDateTime=activity.startDateTime,
                endDateTime=activity.endDateTime,
                assignedWorkerIds=[],
            )

            assigned_count = 0
            for worker_id in worker_ids:
                var_key = (worker_id, activity.id)
                if var_key in assignment_vars and solver.Value(assignment_vars[var_key]) == 1:
                    filled.assignedWorkerIds.append(worker_id)
                    assigned_count += 1

            filled_activities.append(filled)

            expected = activity.constraintData.expectedNbWorker
            fill_rate = assigned_count / expected if expected is not None and expected > 0 else 1.0
            fill_color = "green" if fill_rate >= 0.9 else "yellow" if fill_rate >= 0.7 else "red"

            table.add_row(
                activity.id,
                str(expected),
                str(assigned_count),
                f"[{fill_color}]{fill_rate:.1%}[/{fill_color}]",
            )

        console.print(table)

    return filled_activities


def create_debug_visualizations(timestamp: str, results: dict, test_data: dict):
    """Create debug visualizations for the scheduling results."""
    # Create timestamped directory for this run
    debug_dir = f"debug_logs/{timestamp}"
    os.makedirs(debug_dir, exist_ok=True)

    # Save the current working directory
    current_dir = os.getcwd()

    try:
        # Change to debug directory
        os.chdir(debug_dir)

        # Create images directory
        os.makedirs("images", exist_ok=True)

        # Generate all visualizations
        visualize_activity_assignments(results)
        visualize_worker_distribution(results, test_data)
        # visualize_skill_level_distribution(results, test_data)
        # visualize_timeline(results)

        console.print(f"[bold green]Debug visualizations saved to {debug_dir}/images/")
    finally:
        # Restore the original working directory
        os.chdir(current_dir)
