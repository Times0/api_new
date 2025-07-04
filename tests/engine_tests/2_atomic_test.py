"""
Might be cool to geenrate the test data programatically instead of using json files.
Because the strucutre of the request might change over time so instead of this :
schedule_request = load_test_data("one_activity_levels.json")
we could have something like this :
schedule_request = generate_test_data(nb_activities=1, nb_workers=5, nb_levels=3)
"""

from datetime import date, datetime
from src.engine.engine import solve_scheduling_problem
import time

from tests.utils import (
    load_test_data,
    get_assignment_count,
    get_activity_assignment_levels,
    assert_scheduling_success,
    get_activity_assignments,
    get_worker_assignments,
    assert_unique_worker_assignments,
)
from pathlib import Path


cwd = Path(__file__).parent.resolve()


### Hard constraints tests ###


# todo
def test_localization():
    pass


def test_spread_workers():
    sr = load_test_data(cwd / "test_data/spread_workers_2activities_1assignment.json")
    res = solve_scheduling_problem(sr.activities, sr.workers, sr.options.constraints)
    assert_scheduling_success(res, expected_count=2)
    assert_unique_worker_assignments(res.assignments, expected_count=2)

    # 4 assignments per activity, 2 activities
    sr = load_test_data(cwd / "test_data/spread_workers_2activities_4assignments.json")
    res = solve_scheduling_problem(sr.activities, sr.workers, sr.options.constraints)
    assert_scheduling_success(res, expected_count=8)
    assert_unique_worker_assignments(res.assignments, expected_count=8)


def test_levels():
    sr = load_test_data(cwd.parent / "test_data/one_activity_levels.json")

    res = solve_scheduling_problem(sr.activities, sr.workers, sr.options.constraints)
    assert_scheduling_success(res, expected_count=7)

    # check that we actually get the good level assignments
    activity_levels = get_activity_assignment_levels(sr, res.assignments)

    activity_id = sr.activities[0].id
    assert activity_levels[activity_id]["LEVEL0"] >= 3
    assert activity_levels[activity_id]["LEVEL2"] >= 1
    assert activity_levels[activity_id]["LEVEL3"] >= 3

    # Check that we have the same result with 99% instead of 100% weight
    sr.options.constraints.skillLevels.weight = 0.99
    res_99 = solve_scheduling_problem(sr.activities, sr.workers, sr.options.constraints)
    assert_scheduling_success(res_99, expected_count=7)


def test_pre_assignments():
    """One task, one assignment, preassigned to a guy we want the engine to not do anything else"""
    sr = load_test_data(cwd / "test_data/pre_assignments.json")

    res = solve_scheduling_problem(sr.activities, sr.workers, sr.options.constraints)
    assert_scheduling_success(res, expected_count=1)

    # Check that we have exactly 1 assignment for the preassigned worker
    preassigned_worker_id = "WORKER#1ca75c88-acd6-4830-8d7d-a4600b3813f7"
    nb_assignments = get_worker_assignments(res.assignments, preassigned_worker_id)
    assert nb_assignments == 1, f"Expected 1 assignment for preassigned worker, got {nb_assignments}"


def test_one_shift_per_day_easy():
    """3 Tasks with expected workers == all workers, expected for the engine to choose and fill only 1 activity"""
    sr = load_test_data(cwd.parent / "test_data/one_shift_per_day_easy.json")

    res = solve_scheduling_problem(sr.activities, sr.workers, sr.options.constraints)
    assert_scheduling_success(res, expected_count=240)

    # Check that we have exactly 1 assignment for the preassigned worker
    activity_ids = [activity.id for activity in sr.activities]
    nb_assignments = [get_activity_assignments(res.assignments, activity_id) for activity_id in activity_ids]

    # Verify that exactly one activity has 236 assignments and others have 0
    assert sum(1 for nb in nb_assignments if nb == 240) == 1, f"Expected exactly 1 activity to have 240 assignments, got {nb_assignments}"
    assert all(nb == 0 or nb == 240 for nb in nb_assignments), f"Expected all other activities to have 0 assignments, got {nb_assignments}"


def test_one_shift_per_day_medium():
    """4 Tasks in different shifts Tasks with expected workers == all workers, expected for the engine to choose the multi day task so it fills 2 activities in stead of just 1"""
    sr = load_test_data(cwd.parent / "test_data/one_shift_per_day_medium.json")

    res = solve_scheduling_problem(sr.activities, sr.workers, sr.options.constraints)
    assert_scheduling_success(res, expected_count=16)

    # Check that we have exactly 1 assignment for the preassigned worker
    activity_ids = [activity.id for activity in sr.activities]
    nb_assignments = [get_activity_assignments(res.assignments, activity_id) for activity_id in activity_ids]

    # Verify that exactly one activity has 236 assignments and others have 0
    assert sum(1 for nb in nb_assignments if nb == 8) == 2, f"Expected exactly 2 activity to have 8 assignments, got {nb_assignments}"
    assert all(nb == 0 or nb == 8 for nb in nb_assignments), f"Expected all other activities to have 0 assignments, got {nb_assignments}"


def test_one_shift_per_day_medium_plus():
    """8 Tasks with expected workers [all_workers - 2, 1 ,1 ,1 ,1 ,1 ,1 , all_workers - 2], expected for the engine to fill all activities fully"""
    sr = load_test_data(cwd.parent / "test_data/one_shift_per_day_medium_plus.json")

    res = solve_scheduling_problem(sr.activities, sr.workers, sr.options.constraints)
    assert_scheduling_success(res, expected_count=18)

    # Check that we have exactly 1 assignment for the preassigned worker
    activity_ids = [activity.id for activity in sr.activities]
    nb_assignments = [get_activity_assignments(res.assignments, activity_id) for activity_id in activity_ids]

    # Verify that exactly one activity has 236 assignments and others have 0
    assert sum(1 for nb in nb_assignments if nb == 6) == 2, f"Expected exactly 2 activity to have 6 assignments, got {nb_assignments}"
    assert all(nb == 1 or nb == 6 for nb in nb_assignments), f"Expected all other activities to have 1 assignments, got {nb_assignments}"


def test_one_activity_availibility():
    """1 Task with expected workers all_workers, expected for the engine to not fill the activity as the worker is on leave"""
    sr = load_test_data(cwd.parent / "test_data/availibility_one_activity.json")

    res = solve_scheduling_problem(sr.activities, sr.workers, sr.options.constraints)
    assert_scheduling_success(res, expected_count=0)


def test_two_activity_availibility():
    """2 Task with expected workers all_workers, expected for the engine to fill one activity where the worker is not on leave"""
    sr = load_test_data(cwd.parent / "test_data/availibility_two_activities.json")

    res = solve_scheduling_problem(sr.activities, sr.workers, sr.options.constraints)
    assert_scheduling_success(res, expected_count=1)

    activity_ids = [activity.id for activity in sr.activities]
    nb_assignments = [get_activity_assignments(res.assignments, activity_id) for activity_id in activity_ids]

    # Verify that exactly one activity has 1 assignments and others have 0
    assert sum(1 for nb in nb_assignments if nb == 1) == 1, f"Expected exactly 1 activity to have 1 assignments, got {nb_assignments}"
    assert all(nb == 1 or nb == 0 for nb in nb_assignments), f"Expected all other activities to have 0 assignments, got {nb_assignments}"

    # Verify that the filled activity is the one on day 14
    filled_activity_idx = nb_assignments.index(1)
    filled_activity = sr.activities[filled_activity_idx]
    filled_activity_date = datetime.fromisoformat(filled_activity.startDateTime.replace("Z", "+00:00")).date()
    assert filled_activity_date == date(2024, 6, 14), f"Expected activity on day 14 to be filled, got activity on {filled_activity_date}"


def test_trainings_availibility_easy():
    """
    1 Task with  expected workers all_workers, expected for the engine to fill 0 activities because the worker is in a training
    """
    sr = load_test_data(cwd.parent / "test_data/trainings_availibility_easy.json")

    res = solve_scheduling_problem(sr.activities, sr.workers, sr.options.constraints)
    assert_scheduling_success(res, expected_count=0)

    # There is only one activity and one worker, and the worker is on training during the activity
    activity_ids = [activity.id for activity in sr.activities]
    nb_assignments = [get_activity_assignments(res.assignments, activity_id) for activity_id in activity_ids]

    assert nb_assignments == [0], f"Expected no assignments due to training, got {nb_assignments}"


### Soft constraints tests ###
