"""Tests for the result export module."""

import pytest
import json
from pathlib import Path
from datetime import datetime

from pocket_agent_cli.utils.result_export import (
    export_results,
    load_results,
    compare_results,
)
from pocket_agent_cli.benchmarks.benchmark_service import (
    BenchmarkSession,
    BenchmarkProblemResult,
    TestResult,
)


class TestExportResults:
    """Tests for export_results function."""

    @pytest.fixture
    def sample_session(self):
        """Create a sample benchmark session for testing."""
        return BenchmarkSession(
            session_id="test_session_001",
            model_id="test-model",
            mode="base",
            start_time=datetime(2024, 1, 1, 10, 0, 0),
            end_time=datetime(2024, 1, 1, 10, 5, 0),
            problems=[
                BenchmarkProblemResult(
                    problem_id=1,
                    start_time=datetime(2024, 1, 1, 10, 0, 0),
                    end_time=datetime(2024, 1, 1, 10, 1, 0),
                    response="def solution():\n    return 42",
                    tool_calls=None,
                    test_results=[
                        TestResult(
                            test_case="assert solution() == 42",
                            passed=True,
                            output="",
                        ),
                        TestResult(
                            test_case="assert solution() != 0",
                            passed=True,
                            output="",
                        ),
                    ],
                    success=True,
                    metrics={
                        "ttft": 150.0,
                        "tps": 45.5,
                        "tokens": 15,
                    },
                ),
                BenchmarkProblemResult(
                    problem_id=2,
                    start_time=datetime(2024, 1, 1, 10, 1, 0),
                    end_time=datetime(2024, 1, 1, 10, 2, 0),
                    response="def add(a, b):\n    return a + b",
                    tool_calls=None,
                    test_results=[
                        TestResult(
                            test_case="assert add(1, 2) == 3",
                            passed=True,
                            output="",
                        ),
                        TestResult(
                            test_case="assert add(0, 0) == 1",
                            passed=False,
                            output="AssertionError",
                        ),
                    ],
                    success=False,
                    metrics={
                        "ttft": 120.0,
                        "tps": 50.0,
                        "tokens": 12,
                    },
                ),
            ],
            aggregate_stats={
                "total_problems": 2,
                "passed_problems": 1,
                "pass_rate": 0.5,
                "total_duration_seconds": 300.0,
                "avg_ttft_ms": 135.0,
                "avg_tps": 47.75,
            },
        )

    def test_export_json(self, temp_results_dir, sample_session):
        """Test exporting results to JSON."""
        output_path = temp_results_dir / "results.json"

        export_results(sample_session, output_path)

        assert output_path.exists()

        with open(output_path) as f:
            data = json.load(f)

        assert data["session_id"] == "test_session_001"
        assert data["model_id"] == "test-model"
        assert data["mode"] == "base"
        assert len(data["problems"]) == 2
        assert "export_metadata" in data
        assert data["export_metadata"]["format_version"] == "1.0"

    def test_export_json_problems(self, temp_results_dir, sample_session):
        """Test that problem data is correctly exported."""
        output_path = temp_results_dir / "results.json"

        export_results(sample_session, output_path)

        with open(output_path) as f:
            data = json.load(f)

        problem1 = data["problems"][0]
        assert problem1["problem_id"] == 1
        assert problem1["success"] is True
        assert len(problem1["test_results"]) == 2
        assert problem1["metrics"]["ttft"] == 150.0

    def test_export_csv(self, temp_results_dir, sample_session):
        """Test exporting results to CSV."""
        output_path = temp_results_dir / "results.csv"

        export_results(sample_session, output_path)

        assert output_path.exists()

        content = output_path.read_text()
        lines = content.strip().split('\n')

        # Check header
        assert "problem_id" in lines[0]
        assert "success" in lines[0]
        assert "ttft_ms" in lines[0]
        assert "tps" in lines[0]

        # Check data rows (2 problems + header = 3 lines)
        assert len(lines) == 3

    def test_export_markdown(self, temp_results_dir, sample_session):
        """Test exporting results to Markdown."""
        output_path = temp_results_dir / "results.md"

        export_results(sample_session, output_path)

        assert output_path.exists()

        content = output_path.read_text()

        assert "# Benchmark Results" in content
        assert "test-model" in content
        assert "base" in content
        assert "## Summary" in content
        assert "## Problem Results" in content
        assert "Problem 1" in content
        assert "Problem 2" in content

    def test_export_markdown_contains_stats(self, temp_results_dir, sample_session):
        """Test that markdown contains statistics."""
        output_path = temp_results_dir / "results.md"

        export_results(sample_session, output_path)

        content = output_path.read_text()

        assert "Total Problems" in content
        assert "Passed" in content
        assert "50.0%" in content  # Pass rate

    def test_export_default_json(self, temp_results_dir, sample_session):
        """Test that unknown extension defaults to JSON."""
        output_path = temp_results_dir / "results.unknown"

        export_results(sample_session, output_path)

        # Should create .json file instead
        json_path = output_path.with_suffix(".json")
        assert json_path.exists()

    def test_export_metadata(self, temp_results_dir, sample_session):
        """Test that export metadata is added."""
        output_path = temp_results_dir / "results.json"

        export_results(sample_session, output_path)

        with open(output_path) as f:
            data = json.load(f)

        assert "export_metadata" in data
        assert "exported_at" in data["export_metadata"]
        assert data["export_metadata"]["tool"] == "pocket-agent-cli"


class TestLoadResults:
    """Tests for load_results function."""

    def test_load_json(self, temp_results_dir):
        """Test loading JSON results."""
        data = {
            "session_id": "test_123",
            "model_id": "test-model",
            "aggregate_stats": {
                "pass_rate": 0.8,
            }
        }

        file_path = temp_results_dir / "results.json"
        with open(file_path, 'w') as f:
            json.dump(data, f)

        loaded = load_results(file_path)

        assert loaded["session_id"] == "test_123"
        assert loaded["model_id"] == "test-model"
        assert loaded["aggregate_stats"]["pass_rate"] == 0.8

    def test_load_nonexistent_file(self, temp_results_dir):
        """Test loading non-existent file raises error."""
        with pytest.raises(FileNotFoundError):
            load_results(temp_results_dir / "nonexistent.json")

    def test_load_preserves_all_data(self, temp_results_dir):
        """Test that all data is preserved during load."""
        data = {
            "session_id": "test_456",
            "custom_field": "custom_value",
            "nested": {
                "deeply": {
                    "nested": "value"
                }
            }
        }

        file_path = temp_results_dir / "results.json"
        with open(file_path, 'w') as f:
            json.dump(data, f)

        loaded = load_results(file_path)

        assert loaded == data


class TestCompareResults:
    """Tests for compare_results function."""

    @pytest.fixture
    def results1(self):
        """First result set for comparison."""
        return {
            "model_id": "model-a",
            "mode": "base",
            "aggregate_stats": {
                "pass_rate": 0.7,
                "total_duration_seconds": 100.0,
                "avg_ttft_ms": 150.0,
                "avg_tps": 40.0,
            }
        }

    @pytest.fixture
    def results2(self):
        """Second result set for comparison."""
        return {
            "model_id": "model-b",
            "mode": "base",
            "aggregate_stats": {
                "pass_rate": 0.8,
                "total_duration_seconds": 80.0,
                "avg_ttft_ms": 120.0,
                "avg_tps": 50.0,
            }
        }

    def test_compare_models(self, results1, results2):
        """Test comparing two results."""
        comparison = compare_results(results1, results2)

        assert comparison["model1"] == "model-a"
        assert comparison["model2"] == "model-b"
        assert comparison["mode"] == "base"

    def test_compare_pass_rate(self, results1, results2):
        """Test pass rate comparison."""
        comparison = compare_results(results1, results2)

        # model-b has 0.8, model-a has 0.7, so change is +0.1
        assert comparison["pass_rate_change"] == pytest.approx(0.1)

    def test_compare_duration(self, results1, results2):
        """Test duration comparison."""
        comparison = compare_results(results1, results2)

        # model-b: 80s, model-a: 100s, change is -20
        assert comparison["duration_change"] == pytest.approx(-20.0)

    def test_compare_ttft(self, results1, results2):
        """Test TTFT comparison."""
        comparison = compare_results(results1, results2)

        # model-b: 120ms, model-a: 150ms, change is -30ms
        assert comparison["ttft_change_ms"] == pytest.approx(-30.0)

        # Percent change: (120-150)/150 * 100 = -20%
        assert comparison["ttft_change_percent"] == pytest.approx(-20.0)

    def test_compare_tps(self, results1, results2):
        """Test TPS comparison."""
        comparison = compare_results(results1, results2)

        # model-b: 50 TPS, model-a: 40 TPS, change is +10
        assert comparison["tps_change"] == pytest.approx(10.0)

        # Percent change: (50-40)/40 * 100 = 25%
        assert comparison["tps_change_percent"] == pytest.approx(25.0)

    def test_compare_missing_stats(self):
        """Test comparison with missing stats."""
        results1 = {
            "model_id": "model-a",
            "mode": "base",
            "aggregate_stats": {}
        }
        results2 = {
            "model_id": "model-b",
            "mode": "base",
            "aggregate_stats": {}
        }

        comparison = compare_results(results1, results2)

        assert comparison["model1"] == "model-a"
        assert comparison["model2"] == "model-b"
        assert comparison["pass_rate_change"] == 0
        assert "ttft_change_ms" not in comparison
        assert "tps_change" not in comparison


class TestRoundTrip:
    """Tests for export/load round-trip."""

    def test_json_round_trip(self, temp_results_dir, sample_benchmark_session):
        """Test that JSON export/load preserves data."""
        output_path = temp_results_dir / "round_trip.json"

        # Export
        export_results(sample_benchmark_session, output_path)

        # Load
        loaded = load_results(output_path)

        # Verify key fields
        assert loaded["session_id"] == sample_benchmark_session.session_id
        assert loaded["model_id"] == sample_benchmark_session.model_id
        assert loaded["mode"] == sample_benchmark_session.mode
        assert len(loaded["problems"]) == len(sample_benchmark_session.problems)

    def test_multiple_exports_same_file(self, temp_results_dir, sample_benchmark_session):
        """Test that exporting to same file overwrites."""
        output_path = temp_results_dir / "overwrite.json"

        # First export
        export_results(sample_benchmark_session, output_path)

        # Modify session
        sample_benchmark_session.session_id = "modified_session"

        # Second export
        export_results(sample_benchmark_session, output_path)

        # Load should have modified version
        loaded = load_results(output_path)
        assert loaded["session_id"] == "modified_session"
