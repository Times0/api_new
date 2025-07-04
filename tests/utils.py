import json
from os import PathLike
from pathlib import Path
from typing import Dict, Tuple, Any

import uuid
from datetime import datetime, timedelta

from src.engine.models import Constraint, ConstraintData, MaxShiftsPerDayConstraint
from src.api.models import (
    ScheduleRequest,
    Worker,
    Activity,
    Constraints,
    AutoSchedulingOptions,
)

# Note: If using Python <3.10, use typing.Union instead of X | Y for unions.

test_data_folder = Path(__file__).parent / "test_data"


def load_test_data(filename: PathLike) -> ScheduleRequest:
    """Load test data from a JSON file and convert it to ScheduleRequest object."""
    file_path = test_data_folder / filename
    data = json.loads(file_path.read_text())
    return ScheduleRequest(**data)


def get_assignment_count(assignments: Dict[Tuple[str, str], bool]) -> int:
    """Count the number of assignments in the filled activities."""
    return sum(1 for assigned in assignments.values() if assigned)


def get_activity_assignment_levels(
    sr: ScheduleRequest, assignments: Dict[Tuple[str, str], bool]
) -> Dict[str, Dict[str, int]]:  # activity_id -> {level -> count}
    """Get the levels of the activity assignments."""
    activity_levels: Dict[str, Dict[str, int]] = {}
    for activity in sr.activities:
        activity_levels[activity.id] = {}
        for worker in sr.workers:
            if assignments.get((worker.id, activity.id), False):
                # Defensive: check skillLevels and constraintData exist
                if (
                    sr.options.constraints is not None
                    and getattr(sr.options.constraints, "skillLevels", None) is not None
                    and getattr(sr.options.constraints.skillLevels, "constraintData", None) is not None
                    and worker.id in sr.options.constraints.skillLevels.constraintData
                    and activity.workstationId in sr.options.constraints.skillLevels.constraintData[worker.id]
                ):
                    level = sr.options.constraints.skillLevels.constraintData[worker.id][activity.workstationId]
                else:
                    level = "LEVEL0"  # fallback if not set
                if level not in activity_levels[activity.id]:
                    activity_levels[activity.id][level] = 0
                activity_levels[activity.id][level] += 1
    return activity_levels


def assert_scheduling_success(res, expected_count=None):
    """Assert that scheduling was successful and optionally check the number of assignments."""
    assert res.success is True, f"Scheduling failed: {res.error_message}"
    if expected_count is not None:
        count = get_assignment_count(res.assignments)
        assert count == expected_count, f"Expected {expected_count} assignments, got {count}"


def get_activity_assignments(assignments: Dict[Tuple[str, str], bool], activity_id: str) -> int:
    """Count the number of assignments for a specific activity."""
    return sum(1 for (_, act_id), assigned in assignments.items() if assigned and act_id == activity_id)


def get_worker_assignments(assignments: Dict[Tuple[str, str], bool], worker_id: str) -> int:
    """Count the number of assignments for a specific worker."""
    return sum(1 for (w_id, _), assigned in assignments.items() if assigned and w_id == worker_id)


def assert_unique_worker_assignments(assignments: Dict[Tuple[str, str], bool], expected_count: int):
    """Assert that exactly expected_count different workers are assigned."""
    assigned_workers = set()
    for (worker_id, _), assigned in assignments.items():
        if assigned:
            assigned_workers.add(worker_id)
    assert len(assigned_workers) == expected_count, f"Expected {expected_count} different workers, got {len(assigned_workers)}"


