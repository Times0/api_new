import json
import os
from datetime import datetime

import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend for server environments
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_results(filename="test_results.json"):
    """Load results from a JSON file."""
    with open(filename, "r") as f:
        data = json.load(f)
    return data


def load_test_data(filename="test_data.json"):
    """Load test data from a JSON file."""
    with open(filename, "r") as f:
        data = json.load(f)
    return data


def parse_datetime(dt_str):
    """Parse datetime string to datetime object."""
    return datetime.fromisoformat(dt_str)


def visualize_activity_assignments(results):
    """Visualize the assignment of workers to activities."""
    activities = results["activities"]

    # Sort activities by start time
    activities.sort(key=lambda a: parse_datetime(a["startDateTime"]))
    # Prepare data for visualization
    activity_ids = [a["id"] for a in activities]
    expected_workers = [a["constraintData"].get("expectedNbWorker", 0) or 0 for a in activities]
    assigned_workers = [len(a["assignedWorkerIds"]) for a in activities]

    # Calculate fill rates
    fill_rates = [assigned / expected if expected > 0 else 1.0 for assigned, expected in zip(assigned_workers, expected_workers)]

    # Create figure
    plt.figure(figsize=(12, 8))

    # Plot expected vs assigned workers
    x = np.arange(len(activity_ids))
    width = 0.35

    plt.bar(x - width / 2, expected_workers, width, label="Expected Workers")
    plt.bar(x + width / 2, assigned_workers, width, label="Assigned Workers")

    # Add fill rate as text
    for i, rate in enumerate(fill_rates):
        plt.text(
            i,
            max(expected_workers[i], assigned_workers[i]) + 0.5,
            f"{rate:.1%}",
            ha="center",
            va="bottom",
        )

    plt.xlabel("Activities")
    plt.ylabel("Number of Workers")
    plt.title("Expected vs Assigned Workers per Activity")
    plt.xticks(x, activity_ids, rotation=45)
    plt.legend()
    plt.tight_layout()

    plt.savefig("images/activity_assignments.png")
    print("Saved activity assignments chart to activity_assignments.png")


def visualize_worker_distribution(results, test_data):
    """Visualize the distribution of workers across activities."""
    activities = results["activities"]
    workers = test_data["workers"]

    # Create worker utilization dictionary
    worker_util = {w["id"]: 0 for w in workers}

    # Count assignments per worker
    for activity in activities:
        for worker_id in activity["assignedWorkerIds"]:
            worker_util[worker_id] += 1

    # Convert to dataframe for easier plotting
    df = pd.DataFrame(
        {
            "worker_id": list(worker_util.keys()),
            "assignments": list(worker_util.values()),
        }
    )

    # Sort by number of assignments
    df = df.sort_values("assignments", ascending=False)

    # Create bar chart
    plt.figure(figsize=(12, 8))
    plt.bar(df["worker_id"], df["assignments"])
    plt.xlabel("Worker ID")
    plt.ylabel("Number of Assignments")
    plt.title("Worker Utilization")
    plt.xticks(rotation=90)
    plt.tight_layout()

    plt.savefig("images/worker_distribution.png")
    print("Saved worker distribution chart to worker_distribution.png")


def visualize_skill_level_distribution(results, test_data):
    """Visualize how workers with different skill levels are distributed."""
    activities = results["activities"]
    worker_workstations = test_data["options"]["constraints"]["skillLevels"]["constraintData"]

    # Initialize counters for each skill level
    level_assignments = {
        "LEVEL0": 0,
        "LEVEL1": 0,
        "LEVEL2": 0,
        "LEVEL3": 0,
        "LEVEL4": 0,
    }

    # For each activity, look up the level of each assigned worker on that workstation
    for activity in activities:
        workstation_id = activity["workstationId"]
        workstation_levels = worker_workstations.get(workstation_id, {})

        # For each worker assigned to this activity
        for worker_id in activity["assignedWorkerIds"]:
            # Find what level this worker is on this workstation
            for level, workers in workstation_levels.items():
                if worker_id in workers:
                    level_assignments[level] += 1
                    break

    # Create bar chart
    levels = list(level_assignments.keys())
    values = list(level_assignments.values())

    plt.figure(figsize=(10, 6))
    plt.bar(levels, values)
    plt.xlabel("Skill Level")
    plt.ylabel("Number of Assignments")
    plt.title("Assignment Distribution by Worker Skill Level")
    plt.tight_layout()

    plt.savefig("images/skill_level_distribution.png")
    print("Saved skill level distribution chart to skill_level_distribution.png")


def visualize_timeline(results):
    """Visualize the schedule as a timeline."""
    activities = results["activities"]
    if len(activities) == 0:
        return

    # Create a dataframe for easier plotting
    data = []
    for activity in activities:
        start = parse_datetime(activity["startDateTime"])
        end = parse_datetime(activity["endDateTime"])
        for worker_id in activity["assignedWorkerIds"]:
            data.append(
                {
                    "worker_id": worker_id,
                    "activity_id": activity["id"],
                    "start": start,
                    "end": end,
                }
            )

    df = pd.DataFrame(data)

    # Sort workers by most assigned
    worker_counts = df["worker_id"].value_counts()
    sorted_workers = worker_counts.index.tolist()

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 10))

    # Plot each assignment as a horizontal bar
    for i, worker_id in enumerate(sorted_workers):
        worker_data = df[df["worker_id"] == worker_id]
        for _, row in worker_data.iterrows():
            ax.barh(
                i,
                (row["end"] - row["start"]).total_seconds() / 3600,
                left=(row["start"] - df["start"].min()).total_seconds() / 3600,
                height=0.5,
                color="blue",
                alpha=0.7,
            )
            ax.text(
                (row["start"] - df["start"].min()).total_seconds() / 3600 + (row["end"] - row["start"]).total_seconds() / 3600 / 2,
                i,
                row["activity_id"],
                ha="center",
                va="center",
                color="white",
                fontsize=8,
            )

    # Set the y-ticks to be the worker IDs
    ax.set_yticks(range(len(sorted_workers)))
    ax.set_yticklabels(sorted_workers)

    # Set the x-axis to be hours from the start
    ax.set_xlabel("Hours from Start")
    ax.set_ylabel("Worker ID")
    ax.set_title("Worker Schedule Timeline")

    plt.tight_layout()
    plt.savefig("images/schedule_timeline.png")
    print("Saved schedule timeline to schedule_timeline.png")


def visualize_results():
    """Visualize all aspects of the scheduling results."""
    timestamp = "20250505_124930"
    results = load_results(f"output_{timestamp}.json")
    test_data = load_test_data(f"input_{timestamp}.json")

    os.makedirs("images", exist_ok=True)

    visualize_activity_assignments(results)
    visualize_worker_distribution(results, test_data)
    # visualize_skill_level_distribution(results, test_data)
    visualize_timeline(results)

    print("All visualizations completed!")


if __name__ == "__main__":
    visualize_results()
