from engine.engine import solve_scheduling_problem

from tests.utils import (
    load_test_data,
    assert_scheduling_success,
)
from pathlib import Path

cwd = Path(__file__).parent.resolve()


def test_simplest():
    """
    Test the simplest scheduling problem with one activity and one expected
    """
    sr = load_test_data(cwd / "test_data/simplest.json")
    res = solve_scheduling_problem(sr.activities, sr.workers, sr.options.constraints)
    assert_scheduling_success(res, expected_count=5)


def test_no_activity():
    """
    Test scheduling with no activities - should succeed with empty assignments
    """
    sr = load_test_data(cwd / "test_data/no_activity.json")
    res = solve_scheduling_problem(sr.activities, sr.workers, sr.options.constraints)
    assert_scheduling_success(res, expected_count=0)


def test_shortage_when_insufficient_workers():
    """
    We expect 10, we have only 2 workers
    """
    sr = load_test_data(cwd.parent / "test_data/shortage_when_insufficient_workers.json")

    res = solve_scheduling_problem(sr.activities, sr.workers, sr.options.constraints)
    assert_scheduling_success(res, expected_count=2)


def test_six_overlapping_activities():
    """
    6 overlapping activities the first 5 have 5 required, and the last one has 7
    """
    sr = load_test_data(cwd.parent / "test_data/six_overlapping_activities.json")
    nb_workers = len(sr.workers)
    res = solve_scheduling_problem(sr.activities, sr.workers, sr.options.constraints)
    assert_scheduling_success(res, expected_count=nb_workers)

    # Assert that all workers are assigned only once in total.
    for worker in sr.workers:
        nb_assignments = sum(1 for (worker_id, _), assigned in res.assignments.items() if assigned and worker_id == worker.id)
        assert nb_assignments == 1
