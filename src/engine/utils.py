from datetime import datetime, timezone
import functools
import time

from ortools.sat.python import cp_model
from engine.models import Activity

PENALTY_SCALE = 1000000  # penalties can be only int that is why we can t scale them to 100 without losing information


def timer(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__} took {end - start:.6f} seconds")
        return result

    return wrapper


@timer
def normalize_penalty_list(
    model: cp_model.CpModel,
    penalty_vars: list[cp_model.IntVar],
    max_possible: int,
    name_prefix: str = "normalized",
) -> cp_model.IntVar:
    assert max_possible >= 0, "Max possible penalty must be non-negative"
    if max_possible == 0:
        # If no penalty is possible, return a constant zero variable
        return model.new_int_var(0, 0, f"{name_prefix}_zero")

    total_penalty = model.new_int_var(0, max_possible, f"{name_prefix}_total")
    model.add(total_penalty == sum(penalty_vars))

    normalized = model.new_int_var(0, PENALTY_SCALE, f"{name_prefix}_scaled")
    model.AddDivisionEquality(normalized, total_penalty * PENALTY_SCALE, max_possible)

    return normalized


def normalize_penalty(
    model: cp_model.CpModel,
    penalty_var: cp_model.IntVar,
    max_possible: int,
    name_prefix: str = "normalized",
) -> cp_model.IntVar:
    assert max_possible >= 0, "Max possible penalty must be non-negative"
    if max_possible == 0:
        # If no penalty is possible, return a constant zero variable
        return model.new_int_var(0, 0, f"{name_prefix}_zero")

    normalized = model.new_int_var(0, PENALTY_SCALE, f"{name_prefix}_scaled")
    model.AddDivisionEquality(normalized, penalty_var * PENALTY_SCALE, max_possible)

    return normalized


def datetime_to_slot(dt_str: str, reference_dt: datetime, time_slot_size: int = 15) -> int:
    # Make both datetime timezone-aware in UTC
    dt = datetime.fromisoformat(dt_str.replace("Z", "+00:00")).astimezone(timezone.utc)
    reference_dt = reference_dt.astimezone(timezone.utc)

    delta = dt - reference_dt
    return int(delta.total_seconds() // (time_slot_size * 60))  # 15-min slots


def get_slot_range(activity, first_day_of_window):
    start = datetime_to_slot(activity.startDateTime, first_day_of_window)
    end = datetime_to_slot(activity.endDateTime, first_day_of_window)
    return start, end


def check_rest_windows(
    model: cp_model.CpModel,
    worker_id: str,
    activities: list[Activity],
    assignment_vars: dict,
    first_day_of_window: datetime,
    window_length: int,
    latest_slot: int,
):
    """Check for valid rest windows in the schedule."""
    rest_window_bools = []

    for rest_start in range(0, latest_slot - window_length + 1):  # step by 1h
        rest_end = rest_start + window_length
        overlapping_vars = []

        for activity in activities:
            if (worker_id, activity.id) not in assignment_vars:
                continue

            start_slot, end_slot = get_slot_range(activity, first_day_of_window)

            # Check overlap
            if end_slot > rest_start and start_slot < rest_end:
                var = assignment_vars[(worker_id, activity.id)]
                overlapping_vars.append(var)

        no_work = model.NewBoolVar(f"rest_window_{worker_id}_{rest_start}")

        if overlapping_vars:
            model.add(sum(overlapping_vars) == 0).only_enforce_if(no_work)
            model.add(sum(overlapping_vars) > 0).only_enforce_if(no_work.Not())
            rest_window_bools.append(no_work)
        else:
            model.add(no_work == 1)
            rest_window_bools.append(no_work)

    return rest_window_bools


