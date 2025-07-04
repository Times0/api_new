import time
from typing import Optional, NamedTuple
import pytest
from engine.engine import solve_scheduling_problem
from utils import generate_performance_test_data
from rich.console import Console
from rich.table import Table
from collections import defaultdict

# Add matplotlib for graphing
import matplotlib.pyplot as plt


class TestConfig(NamedTuple):
    """Configuration for a performance test case."""

    name: str
    nb_activities: int
    nb_workers: int


class TestResult(NamedTuple):
    """Result of a performance test."""

    config: TestConfig
    elapsed_time: float
    penalty: Optional[float]
    wall_time: Optional[float]


class PerformanceTestRunner:
    """Manages performance test execution and result collection."""

    def __init__(self):
        self.results: list[TestResult] = []
        self.console = Console()

    def _extract_value(self, res, keys: list[str]) -> Optional[float]:
        """Extract a value from solver result using multiple possible keys."""
        stats = getattr(res, "statistics", {})
        if isinstance(stats, dict):
            for key in keys:
                if (value := stats.get(key)) is not None:
                    return value

        # Try direct attributes as fallback
        for key in keys:
            if (value := getattr(res, key, None)) is not None:
                return value
        return None

    def run_test(self, config: TestConfig) -> TestResult:
        """Run a single performance test."""
        print(f"Running {config.name}: {config.nb_activities} activities, {config.nb_workers} workers")

        # Generate test data and run
        sr = generate_performance_test_data(config.nb_activities, config.nb_workers)
        start = time.time()
        res = solve_scheduling_problem(sr.activities, sr.workers, sr.options.constraints)
        elapsed = time.time() - start

        # Extract results
        penalty = self._extract_value(res, ["penalty", "final_penalty"])
        wall_time = self._extract_value(res, ["wall_time", "wallTime"])

        result = TestResult(config, elapsed, penalty, wall_time)
        self.results.append(result)

        print(f"Completed in {elapsed:.2f}s (penalty: {penalty}, wall_time: {wall_time})")
        return result

    def run_all(self, configs: list[TestConfig]) -> list[TestResult]:
        """Run all performance tests."""
        print(f"Starting {len(configs)} performance tests...")
        for config in configs:
            self.run_test(config)
        print("All tests completed!")
        return self.results

    def _get_color(self, value: Optional[float], thresholds: list[tuple]) -> str:
        """Get color based on value and thresholds."""
        if value is None:
            return "green"
        for threshold, color in thresholds:
            if value < threshold:
                return color
        return thresholds[-1][1]

    def print_results(self):
        """Print formatted results table."""
        if not self.results:
            print("No results to display.")
            return

        table = Table(title="Performance Test Results", show_lines=True)
        for col in ["Test", "Activities", "Workers", "Time (s)", "Wall Time (s)", "Penalty"]:
            table.add_column(col, justify="center")

        time_thresholds = [(30, "green"), (60, "yellow"), (120, "orange1"), (180, "orange2"), (float("inf"), "red")]

        for result in self.results:
            config = result.config
            time_color = self._get_color(result.elapsed_time, time_thresholds)
            penalty_color = "red" if result.penalty and (result.penalty > 0 or result.penalty == -1) else "green"

            table.add_row(
                config.name,
                str(config.nb_activities),
                str(config.nb_workers),
                f"[{time_color}]{result.elapsed_time:.2f}[/{time_color}]",
                f"{result.wall_time:.2f}" if result.wall_time else "-",
                f"[{penalty_color}]{result.penalty:.2f}[/{penalty_color}]" if result.penalty is not None else "-",
            )

        self.console.print(table)

    def plot_time_graph(self):
        """Plot a graph showing the increase of time based on workers and activities."""
        if not self.results:
            print("No results to plot.")
            return

        # Group results by fixed activities and fixed workers
        activities_groups = defaultdict(list)
        workers_groups = defaultdict(list)

        for result in self.results:
            activities_groups[result.config.nb_activities].append(result)
            workers_groups[result.config.nb_workers].append(result)

        # Only plot for the activities values that are actually used as "fixed activities" in DEFAULT_CONFIGS
        # That is, only plot for activities=100 (the fixed activities in the first sweep)
        fixed_activities_values = set()
        for config in DEFAULT_CONFIGS:
            if config.nb_activities == 100:
                fixed_activities_values.add(config.nb_activities)
        # If none found, fallback to all activities
        if not fixed_activities_values:
            fixed_activities_values = set(activities_groups.keys())

        # Only plot for the workers values that are actually used as "fixed workers" in DEFAULT_CONFIGS
        # That is, only plot for workers=100 (the fixed workers in the second sweep)
        fixed_workers_values = set()
        for config in DEFAULT_CONFIGS:
            if config.nb_workers == 100:
                fixed_workers_values.add(config.nb_workers)
        # If none found, fallback to all workers
        if not fixed_workers_values:
            fixed_workers_values = set(workers_groups.keys())

        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        max_time = 0
        for activities in sorted(fixed_activities_values):
            group = activities_groups[activities]
            group_sorted = sorted(group, key=lambda r: r.config.nb_workers)
            workers = [r.config.nb_workers for r in group_sorted]
            times = [r.elapsed_time for r in group_sorted]
            if times:
                max_time = max(max_time, max(times))
            plt.plot(workers, times, marker="o", label=f"{activities} activities")
        plt.xlabel("Number of Workers")
        plt.ylabel("Elapsed Time (s)")
        plt.title("Time vs Workers (for fixed activities)")
        plt.legend()
        # Use linear scale, and set y-axis limit to a reasonable value
        if max_time > 0:
            plt.ylim(0, max_time * 1.1)

        plt.subplot(1, 2, 2)
        max_time2 = 0
        for workers in sorted(fixed_workers_values):
            group = workers_groups[workers]
            group_sorted = sorted(group, key=lambda r: r.config.nb_activities)
            activities = [r.config.nb_activities for r in group_sorted]
            times = [r.elapsed_time for r in group_sorted]
            if times:
                max_time2 = max(max_time2, max(times))
            plt.plot(activities, times, marker="o", label=f"{workers} workers")
        plt.xlabel("Number of Activities")
        plt.ylabel("Elapsed Time (s)")
        plt.title("Time vs Activities (for fixed workers)")
        plt.legend()
        # Use linear scale, and set y-axis limit to a reasonable value
        if max_time2 > 0:
            plt.ylim(0, max_time2 * 1.1)

        plt.tight_layout()
        plt.show()


def create_config(activities: int, workers: int) -> TestConfig:
    """Create a test configuration with automatic naming."""
    return TestConfig(f"{activities}x{workers}", activities, workers)


# Default test configurations
# 1. Fix activities at 100, increase workers (start at 10) until activities * workers >= 10000
# 2. Fix workers at 100, increase activities (start at 10) until activities * workers >= 10000

DEFAULT_CONFIGS = []

# Use exponential (multiplicative) increments for broader coverage

# Use variable max_value
max_value = 100000

# Fix activities at 100, increase workers exponentially
activities_fixed = 100
workers = 10
while activities_fixed * workers <= max_value:
    DEFAULT_CONFIGS.append(create_config(activities_fixed, workers))
    if workers < 100:
        workers *= 2
    elif workers < 100000:
        workers *= 2
    else:
        workers *= 2
# Add the exact max_value config if not already present
if activities_fixed * workers > max_value and activities_fixed * (workers // 2) < max_value:
    DEFAULT_CONFIGS.append(create_config(activities_fixed, max_value // activities_fixed))

# Fix workers at 100, increase activities exponentially
workers_fixed = 100
activities = 10
while activities * workers_fixed <= max_value:
    DEFAULT_CONFIGS.append(create_config(activities, workers_fixed))
    if activities < 100:
        activities *= 2
    elif activities < 100000:
        activities *= 2
    else:
        activities *= 2
# Add the exact max_value config if not already present
if activities * workers_fixed > max_value and (activities // 2) * workers_fixed < max_value:
    DEFAULT_CONFIGS.append(create_config(max_value // workers_fixed, workers_fixed))


@pytest.mark.parametrize("config", DEFAULT_CONFIGS)
def test_performance(config: TestConfig):
    """Parametrized performance test for all configurations."""
    runner = PerformanceTestRunner()
    runner.run_test(config)


def run_performance_suite(configs: Optional[list[TestConfig]] = None, plot: bool = False) -> PerformanceTestRunner:
    """Run the complete performance test suite and return results."""
    configs = configs or DEFAULT_CONFIGS
    runner = PerformanceTestRunner()
    runner.run_all(configs)
    runner.print_results()
    if plot:
        runner.plot_time_graph()
    return runner


if __name__ == "__main__":
    # Run all tests and show the graph
    run_performance_suite(DEFAULT_CONFIGS, plot=True)
