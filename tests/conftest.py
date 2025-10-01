"""Shared pytest fixtures for pocket-agent-cli tests."""

import pytest
import tempfile
import json
from pathlib import Path
from datetime import datetime
from unittest.mock import MagicMock, patch
from typing import Dict, Any, List


# ============================================================================
# Directory and Path Fixtures
# ============================================================================

@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def temp_data_dir(temp_dir):
    """Create a temporary data directory with sample dataset."""
    data_dir = temp_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


@pytest.fixture
def temp_results_dir(temp_dir):
    """Create a temporary results directory."""
    results_dir = temp_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    return results_dir


# ============================================================================
# Sample Data Fixtures
# ============================================================================

@pytest.fixture
def sample_problems() -> List[Dict[str, Any]]:
    """Sample MBPP-style problems for testing."""
    return [
        {
            "task_id": 1,
            "text": "Write a function to find the minimum cost path.",
            "code": "def min_cost(cost, m, n):\n    return cost[m][n]",
            "test_list": [
                "assert min_cost([[1, 2], [3, 4]], 1, 1) == 4",
                "assert min_cost([[1]], 0, 0) == 1",
            ]
        },
        {
            "task_id": 2,
            "text": "Write a function to add two numbers.",
            "code": "def add(a, b):\n    return a + b",
            "test_list": [
                "assert add(1, 2) == 3",
                "assert add(0, 0) == 0",
            ]
        },
        {
            "task_id": 3,
            "text": "Write a function to check if a number is prime.",
            "code": "def is_prime(n):\n    if n < 2:\n        return False\n    for i in range(2, n):\n        if n % i == 0:\n            return False\n    return True",
            "test_list": [
                "assert is_prime(2) == True",
                "assert is_prime(4) == False",
                "assert is_prime(7) == True",
            ]
        }
    ]


@pytest.fixture
def sample_dataset_file(temp_data_dir, sample_problems):
    """Create a sample dataset file."""
    dataset_path = temp_data_dir / "mbpp_sample.json"
    with open(dataset_path, 'w') as f:
        json.dump(sample_problems, f, indent=2)
    return dataset_path


@pytest.fixture
def sample_humaneval_problems() -> List[Dict[str, Any]]:
    """Sample HumanEval-style problems for testing."""
    return [
        {
            "task_id": "HumanEval/0",
            "prompt": "def has_close_elements(numbers: List[float], threshold: float) -> bool:\n    \"\"\"Check if any two numbers are closer than threshold.\"\"\"\n",
            "canonical_solution": "    for i, n1 in enumerate(numbers):\n        for n2 in numbers[i+1:]:\n            if abs(n1 - n2) < threshold:\n                return True\n    return False",
            "test": "def check(candidate):\n    assert candidate([1.0, 2.0, 3.0], 0.5) == False\n    assert candidate([1.0, 2.0, 2.2], 0.5) == True\n",
            "entry_point": "has_close_elements"
        },
        {
            "task_id": "HumanEval/1",
            "prompt": "def add(a: int, b: int) -> int:\n    \"\"\"Add two integers.\"\"\"\n",
            "canonical_solution": "    return a + b",
            "test": "def check(candidate):\n    assert candidate(1, 2) == 3\n    assert candidate(0, 0) == 0\n",
            "entry_point": "add"
        }
    ]


# ============================================================================
# Mock Model Fixtures
# ============================================================================

@pytest.fixture
def mock_model():
    """Create a mock Model object."""
    model = MagicMock()
    model.id = "test-model"
    model.name = "Test Model"
    model.architecture = "llama"
    model.quantization = "Q4_K_M"
    model.downloaded = True
    model.path = Path("/tmp/test_model.gguf")
    model.is_downloaded.return_value = True
    return model


@pytest.fixture
def mock_model_gemma():
    """Create a mock Gemma model."""
    model = MagicMock()
    model.id = "gemma-test"
    model.name = "Gemma Test"
    model.architecture = "gemma"
    model.quantization = "Q4_K_M"
    model.downloaded = True
    model.path = Path("/tmp/gemma_test.gguf")
    model.is_downloaded.return_value = True
    return model


@pytest.fixture
def mock_model_qwen():
    """Create a mock Qwen model."""
    model = MagicMock()
    model.id = "qwen-test"
    model.name = "Qwen Test"
    model.architecture = "qwen"
    model.quantization = "Q4_K_M"
    model.downloaded = True
    model.path = Path("/tmp/qwen_test.gguf")
    model.is_downloaded.return_value = True
    return model


# ============================================================================
# Mock Service Fixtures
# ============================================================================

@pytest.fixture
def mock_inference_service(mock_model):
    """Create a mock InferenceService."""
    service = MagicMock()
    service.current_model = mock_model
    service.temperature = 0.7

    # Mock generate method
    def mock_generate(messages, stream=True, **kwargs):
        yield {
            "token": "def solution():\n    return 42",
            "finish_reason": "stop",
            "metrics": {
                "ttft": 100.0,
                "tps": 50.0,
                "tokens": 10,
            }
        }

    service.generate.side_effect = mock_generate
    return service


@pytest.fixture
def mock_tool_executor():
    """Create a mock ToolExecutor."""
    executor = MagicMock()
    executor.sandbox_dir = Path("/tmp/sandbox")

    async def mock_run_python_code(code):
        # Simple mock that returns empty string (test passed)
        if "assert" in code and "False" in code:
            return "AssertionError"
        return ""

    executor._run_python_code = mock_run_python_code
    return executor


# ============================================================================
# Benchmark Result Fixtures
# ============================================================================

@pytest.fixture
def sample_test_result():
    """Create a sample TestResult."""
    from pocket_agent_cli.benchmarks.benchmark_service import TestResult
    return TestResult(
        test_case="assert add(1, 2) == 3",
        passed=True,
        output="Test passed",
    )


@pytest.fixture
def sample_problem_result():
    """Create a sample BenchmarkProblemResult."""
    from pocket_agent_cli.benchmarks.benchmark_service import BenchmarkProblemResult, TestResult

    return BenchmarkProblemResult(
        problem_id=1,
        start_time=datetime(2024, 1, 1, 10, 0, 0),
        end_time=datetime(2024, 1, 1, 10, 0, 5),
        response="def solution():\n    return 42",
        tool_calls=None,
        test_results=[
            TestResult(test_case="assert solution() == 42", passed=True, output=""),
        ],
        success=True,
        metrics={
            "ttft": 100.0,
            "tps": 50.0,
            "tokens": 10,
        },
    )


@pytest.fixture
def sample_benchmark_session(sample_problem_result):
    """Create a sample BenchmarkSession."""
    from pocket_agent_cli.benchmarks.benchmark_service import BenchmarkSession

    return BenchmarkSession(
        session_id="test_session_123",
        model_id="test-model",
        mode="base",
        start_time=datetime(2024, 1, 1, 10, 0, 0),
        end_time=datetime(2024, 1, 1, 10, 1, 0),
        problems=[sample_problem_result],
        aggregate_stats={
            "total_problems": 1,
            "passed_problems": 1,
            "pass_rate": 1.0,
            "total_duration_seconds": 60.0,
        },
    )


# ============================================================================
# Configuration Fixtures
# ============================================================================

@pytest.fixture
def sample_inference_config():
    """Create a sample InferenceConfig."""
    from pocket_agent_cli.config import InferenceConfig
    return InferenceConfig(
        temperature=0.7,
        max_tokens=2048,
        top_p=0.9,
        context_length=4096,
    )


@pytest.fixture
def sample_benchmark_config(temp_results_dir):
    """Create a sample BenchmarkConfig."""
    from pocket_agent_cli.config import BenchmarkConfig
    return BenchmarkConfig(
        model_name="test-model",
        mode="base",
        num_samples=1,
        temperature=0.7,
        output_dir=temp_results_dir,
    )


# ============================================================================
# Environment Patches
# ============================================================================

@pytest.fixture
def mock_env_vars(temp_dir, monkeypatch):
    """Set up mock environment variables for testing."""
    monkeypatch.setenv("POCKET_AGENT_HOME", str(temp_dir))
    monkeypatch.setenv("POCKET_AGENT_DATA_DIR", str(temp_dir / "data"))
    monkeypatch.setenv("POCKET_AGENT_RESULTS_DIR", str(temp_dir / "results"))
    monkeypatch.setenv("DISABLE_DOCKER", "true")
    return temp_dir
