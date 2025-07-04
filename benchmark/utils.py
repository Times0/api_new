from engine.models import Activity, Worker, Constraints
from api.models import ScheduleRequest, AutoSchedulingOptions
from datetime import datetime, timedelta
import uuid


def generate_performance_test_data(nb_activities: int, nb_workers: int) -> ScheduleRequest:
    """
    Generate ScheduleRequest with the given number of activities, workers, and workstations, with minimal constraints.
    Each activity is assigned to a randomly chosen workstation from the created set.
    """
    import random

    # Generate workers
    workers = [Worker(id=f"WORKER#{uuid.uuid4()}", name=f"Worker {i}") for i in range(nb_workers)]

    # Generate workstations
    workstation_ids = [f"WORKSTATION#{uuid.uuid4()}" for _ in range(50)]

    # Generate activities, randomly assigning each to a workstation
    base_time = datetime(2024, 6, 13, 8, 0, 0)
    activities = []
    for i in range(nb_activities):
        activity_id = f"ACTIVITY#{uuid.uuid4()}"
        workstation_id = random.choice(workstation_ids)
        # Ensure activities do not overlap: each starts after the previous one ends
        start = base_time + timedelta(hours=8 * i)
        end = start + timedelta(hours=8)
        activities.append(
            Activity(
                id=activity_id,
                workstationId=workstation_id,
                # Pass constraintData as a dict, not as a ConstraintData instance, to avoid Pydantic validation error
                constraintData={"expectedNbWorker": 1, "skillLevelRequirements": [], "strainLevel": None},
                startDateTime=start.isoformat() + "Z",
                endDateTime=end.isoformat() + "Z",
            )
        )
    # Minimal constraints
    # Use the minimal set of constraints, all weights set to 0 and constraintData set to None or empty dict as appropriate.
    constraints = Constraints(
        spreadWorkers={"weight": 0.0, "constraintData": None},
        localization={"weight": 0.0, "constraintData": None},
        preAssignedWorkers={"weight": 0.0, "constraintData": {}},
        availability={"weight": 0.0, "constraintData": {}},
        workerLocalization={"weight": 0.0, "constraintData": None},
        skillLevels={"weight": 0.0, "constraintData": {}},
        maxShiftsPerDay={
            "weight": 0.0,
            "constraintData": {
                "workerActivitiesMatrix": {},
                "shiftTransitionMatrix": {},
                "shiftIds": [],
                "scheduleDays": [],
            },
        },
        trainings={"weight": 0.0, "constraintData": {}},
        legalHours={"weight": 0.0, "constraintData": {"workingHoursDay": 8, "workingHoursWeek": 40, "restHoursWeek": 0, "weeklyActivities": []}},
    )

    options = AutoSchedulingOptions(constraints=constraints)
    return ScheduleRequest(activities=activities, workers=workers, options=options)
