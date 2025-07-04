import pytest
import json
from fastapi.testclient import TestClient
from api.main import app
from tests.utils import load_test_data

client = TestClient(app)


def test_root_endpoint():
    """Test the root GET endpoint returns the expected message."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Hello World"}


def test_auto_schedule_success_one_activity_easy():
    """Test auto-schedule endpoint with valid data from one_activity_easy.json."""
    schedule_request = load_test_data("one_activity_easy.json")

    response = client.post("/auto-schedule", json=schedule_request.model_dump())

    assert response.status_code == 200
    data = response.json()

    # Verify response structure
    assert "activities" in data
    assert "score" in data
    assert isinstance(data["activities"], list)
    assert isinstance(data["score"], (int, float))

    # Verify we got filled activities (should have assignments)
    assert len(data["activities"]) > 0
    for activity in data["activities"]:
        assert "id" in activity
        assert "assignedWorkerIds" in activity


def test_auto_schedule_success_one_activity_levels():
    """Test auto-schedule endpoint with valid data from one_activity_levels.json."""
    schedule_request = load_test_data("one_activity_levels.json")

    response = client.post("/auto-schedule", json=schedule_request.model_dump())

    assert response.status_code == 200
    data = response.json()

    # Verify response structure
    assert "activities" in data
    assert "score" in data
    assert isinstance(data["activities"], list)
    assert isinstance(data["score"], (int, float))

    # Should have filled activities
    assert len(data["activities"]) > 0


def test_auto_schedule_success_six_overlapping_activities():
    """Test auto-schedule endpoint with complex overlapping activities."""
    schedule_request = load_test_data("six_overlapping_activities.json")

    response = client.post("/auto-schedule", json=schedule_request.model_dump())

    assert response.status_code == 200
    data = response.json()

    # Verify response structure
    assert "activities" in data
    assert "score" in data
    assert isinstance(data["activities"], list)
    assert isinstance(data["score"], (int, float))

    # Should have multiple filled activities
    assert len(data["activities"]) > 1


def test_auto_schedule_invalid_json():
    """Test auto-schedule endpoint with invalid JSON data."""
    invalid_data = {"invalid": "data"}

    response = client.post("/auto-schedule", json=invalid_data)

    assert response.status_code == 422  # Validation error


def test_auto_schedule_empty_activities():
    """Test auto-schedule endpoint with empty activities list."""
    data = {"activities": [], "workers": [{"id": "worker1", "name": "Test Worker"}], "options": {"constraints": None}}

    response = client.post("/auto-schedule", json=data)

    assert response.status_code == 400  # Should fail validation
    assert "No activities provided" in response.json()["detail"]


def test_auto_schedule_empty_workers():
    """Test auto-schedule endpoint with empty workers list."""
    data = {
        "activities": [{"id": "activity1", "name": "Test Activity", "workstationId": "workstation1", "constraintData": {"expectedNbWorker": 1}}],
        "workers": [],
        "options": {"constraints": None},
    }

    response = client.post("/auto-schedule", json=data)

    assert response.status_code == 422  # Should fail validation


def test_auto_schedule_missing_required_fields():
    """Test auto-schedule endpoint with missing required fields."""
    incomplete_data = {
        "activities": [
            {
                "id": "activity1"
                # Missing required fields
            }
        ]
    }

    response = client.post("/auto-schedule", json=incomplete_data)

    assert response.status_code == 422  # Validation error


def test_auto_schedule_response_schema():
    """Test that the auto-schedule response matches the expected schema."""
    schedule_request = load_test_data("one_activity_easy.json")

    response = client.post("/auto-schedule", json=schedule_request.model_dump())

    assert response.status_code == 200
    data = response.json()

    # Verify the response matches Schedule model structure
    assert "activities" in data
    assert "score" in data

    # Each activity should be a FilledActivity
    for activity in data["activities"]:
        assert "id" in activity
        assert "assignedWorkerIds" in activity
        assert isinstance(activity["assignedWorkerIds"], list)
