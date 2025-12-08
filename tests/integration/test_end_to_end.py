"""End-to-end integration tests for pocket-agent-cli.

These tests verify the full pipeline works together:
- Dataset loading -> Prompt generation -> Test execution

Note: Tests that require actual model inference are marked with
@pytest.mark.slow and should be run separately.
"""

import pytest
import asyncio
from pathlib import Path
from unittest.mock import Mock, AsyncMock

from pocket_agent_cli.datasets import DatasetRegistry, Problem
from pocket_agent_cli.benchmarks.benchmark_service import BenchmarkService, TestResult
from pocket_agent_cli.config import BenchmarkConfig, BENCHMARK_MODES


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def data_dir():
    """Get the actual package data directory."""
    return Path(__file__).parent.parent.parent / "pocket_agent_cli" / "data"


@pytest.fixture
def mock_inference_service():
    """Create a mock inference service."""
    mock = Mock()
    mock.current_model = Mock()
    mock.current_model.id = 'test-model'
    mock.current_model.architecture = 'llama'
    return mock


# ============================================================================
# Test: Dataset to BenchmarkService Integration
# ============================================================================

class TestDatasetBenchmarkIntegration:
    """Tests for dataset and benchmark service integration."""

    def test_mbpp_loads_into_benchmark_service(self, mock_inference_service, data_dir):
        """Test MBPP dataset loads correctly into BenchmarkService."""
        config = BenchmarkConfig(model_name='test-model', problems_limit=5)
        service = BenchmarkService(
            inference_service=mock_inference_service,
            config=config,
            dataset_name='mbpp',
            data_dir=data_dir
        )

        assert service.dataset_name == 'mbpp'
        assert len(service.problems) == 5
        assert all(isinstance(p, Problem) for p in service.problems)

    def test_humaneval_loads_into_benchmark_service(self, mock_inference_service, data_dir):
        """Test HumanEval dataset loads correctly into BenchmarkService."""
        config = BenchmarkConfig(model_name='test-model', problems_limit=5)
        service = BenchmarkService(
            inference_service=mock_inference_service,
            config=config,
            dataset_name='humaneval',
            data_dir=data_dir
        )

        assert service.dataset_name == 'humaneval'
        assert len(service.problems) == 5
        assert all(isinstance(p, Problem) for p in service.problems)


# ============================================================================
# Test: Problem to Prompt Integration
# ============================================================================

class TestProblemPromptIntegration:
    """Tests for problem to prompt generation integration."""

    def test_mbpp_problem_generates_valid_prompt(self, mock_inference_service, data_dir):
        """Test MBPP problem generates a valid prompt."""
        config = BenchmarkConfig(model_name='test-model', problems_limit=1)
        service = BenchmarkService(
            inference_service=mock_inference_service,
            config=config,
            dataset_name='mbpp',
            data_dir=data_dir
        )

        problem = service.problems[0]
        problem_dict = service._problem_to_dict(problem)
        mode = BENCHMARK_MODES['base']

        prompt = service._prepare_problem_prompt(problem_dict, mode)

        assert len(prompt) > 0
        assert problem.prompt in prompt or "test cases" in prompt.lower()

    def test_humaneval_problem_generates_valid_prompt(self, mock_inference_service, data_dir):
        """Test HumanEval problem generates a valid prompt."""
        config = BenchmarkConfig(model_name='test-model', problems_limit=1)
        service = BenchmarkService(
            inference_service=mock_inference_service,
            config=config,
            dataset_name='humaneval',
            data_dir=data_dir
        )

        problem = service.problems[0]
        problem_dict = service._problem_to_dict(problem)
        mode = BENCHMARK_MODES['base']

        prompt = service._prepare_problem_prompt(problem_dict, mode)

        assert len(prompt) > 0
        assert "def " in prompt  # HumanEval prompts contain function signatures


# ============================================================================
# Test: Code Execution Integration
# ============================================================================

class TestCodeExecutionIntegration:
    """Tests for code execution integration."""

    @pytest.mark.asyncio
    async def test_mbpp_canonical_solution_passes_tests(self, mock_inference_service, data_dir):
        """Test MBPP canonical solution passes its own tests."""
        config = BenchmarkConfig(model_name='test-model', problems_limit=3)
        service = BenchmarkService(
            inference_service=mock_inference_service,
            config=config,
            dataset_name='mbpp',
            data_dir=data_dir
        )

        for problem in service.problems:
            # Run canonical solution against test cases
            full_code = problem.canonical_solution

            # Test locally (not through tool executor)
            for test_case in problem.test_cases:
                test_code = f"{full_code}\n\n{test_case}"
                try:
                    exec(test_code, {})
                except AssertionError:
                    pytest.fail(f"Problem {problem.task_id} canonical solution failed: {test_case}")

    @pytest.mark.asyncio
    async def test_humaneval_canonical_solution_passes_check(self, mock_inference_service, data_dir):
        """Test HumanEval canonical solution passes check() function."""
        config = BenchmarkConfig(model_name='test-model', problems_limit=3)
        service = BenchmarkService(
            inference_service=mock_inference_service,
            config=config,
            dataset_name='humaneval',
            data_dir=data_dir
        )

        for problem in service.problems:
            full_code = problem.prompt + problem.canonical_solution
            original_test = problem.metadata.get('original_test', '')

            test_code = f"{full_code}\n{original_test}\ncheck({problem.entry_point})"

            try:
                exec(test_code, {})
            except AssertionError:
                pytest.fail(f"Problem {problem.task_id} canonical solution failed check()")


# ============================================================================
# Test: Full Pipeline (Mocked Inference)
# ============================================================================

class TestFullPipelineMocked:
    """Tests for the full pipeline with mocked inference."""

    @pytest.mark.asyncio
    async def test_mbpp_full_pipeline_with_correct_code(self, mock_inference_service, data_dir):
        """Test full MBPP pipeline when model returns correct code."""
        config = BenchmarkConfig(model_name='test-model', problems_limit=1)
        service = BenchmarkService(
            inference_service=mock_inference_service,
            config=config,
            dataset_name='mbpp',
            data_dir=data_dir
        )

        # Get the first problem and its canonical solution
        problem = service.problems[0]

        # Mock the tool executor to use real Python
        service.tool_executor._create_sandbox()

        # Run tests with the canonical solution
        problem_dict = service._problem_to_dict(problem)
        results = await service._run_tests(
            problem.canonical_solution,
            problem_dict["test_list"],
            problem_dict
        )

        # Should pass all tests
        assert all(r.passed for r in results), f"Some tests failed: {[r for r in results if not r.passed]}"

    @pytest.mark.asyncio
    async def test_humaneval_full_pipeline_with_correct_code(self, mock_inference_service, data_dir):
        """Test full HumanEval pipeline when model returns correct code."""
        config = BenchmarkConfig(model_name='test-model', problems_limit=1)
        service = BenchmarkService(
            inference_service=mock_inference_service,
            config=config,
            dataset_name='humaneval',
            data_dir=data_dir
        )

        # Get the first problem and its canonical solution
        problem = service.problems[0]
        full_code = problem.prompt + problem.canonical_solution

        # Mock the tool executor to use real Python
        service.tool_executor._create_sandbox()

        # Run tests with the canonical solution
        problem_dict = service._problem_to_dict(problem)
        results = await service._run_tests(
            full_code,
            problem_dict["test_list"],
            problem_dict
        )

        # Should pass all tests
        assert all(r.passed for r in results), f"Some tests failed: {[r for r in results if not r.passed]}"


# ============================================================================
# Test: Dataset Switching
# ============================================================================

class TestDatasetSwitching:
    """Tests for switching between datasets."""

    def test_switch_from_mbpp_to_humaneval(self, mock_inference_service, data_dir):
        """Test switching from MBPP to HumanEval."""
        config = BenchmarkConfig(model_name='test-model', problems_limit=2)

        # Start with MBPP
        service = BenchmarkService(
            inference_service=mock_inference_service,
            config=config,
            dataset_name='mbpp',
            data_dir=data_dir
        )

        mbpp_problem = service.problems[0]
        assert not service._is_humaneval_problem(service._problem_to_dict(mbpp_problem))

        # Create new service with HumanEval
        service = BenchmarkService(
            inference_service=mock_inference_service,
            config=config,
            dataset_name='humaneval',
            data_dir=data_dir
        )

        he_problem = service.problems[0]
        assert service._is_humaneval_problem(service._problem_to_dict(he_problem))

    def test_both_datasets_have_different_problem_formats(self, mock_inference_service, data_dir):
        """Test that MBPP and HumanEval have different problem formats."""
        config = BenchmarkConfig(model_name='test-model', problems_limit=1)

        # Load MBPP
        mbpp_service = BenchmarkService(
            inference_service=mock_inference_service,
            config=config,
            dataset_name='mbpp',
            data_dir=data_dir
        )
        mbpp_problem = mbpp_service.problems[0]

        # Load HumanEval
        he_service = BenchmarkService(
            inference_service=mock_inference_service,
            config=config,
            dataset_name='humaneval',
            data_dir=data_dir
        )
        he_problem = he_service.problems[0]

        # MBPP has natural language prompt
        assert "def " not in mbpp_problem.prompt

        # HumanEval has function signature in prompt
        assert "def " in he_problem.prompt
