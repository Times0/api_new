from engine.scheduler import SchedulingEngine
from engine.models import Activity, Worker, Constraints
from api.models import Schedule, ScheduleRequest


def test_basic_assertion():
    assert True


def test_import_engine_models():
    from engine.models import Activity, ConstraintData

    assert Activity is not None
    assert ConstraintData is not None


def test_import_api_models():
    from api.models import ScheduleRequest, Schedule

    assert ScheduleRequest is not None
    assert Schedule is not None


def test_engine_imports():
    """Test that engine imports work correctly"""
    assert SchedulingEngine is not None
    assert Activity is not None
    assert Worker is not None
    assert Constraints is not None


def test_api_imports():
    """Test that API imports work correctly"""
    assert Schedule is not None
    assert ScheduleRequest is not None


def test_scheduler_instantiation():
    """Test that scheduler can be instantiated"""
    scheduler = SchedulingEngine()
    assert scheduler is not None
