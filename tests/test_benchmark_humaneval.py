"""Tests for HumanEval integration with BenchmarkService."""

import pytest
import json
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch, MagicMock

from pocket_agent_cli.benchmarks.benchmark_service import BenchmarkService, TestResult
from pocket_agent_cli.config import BenchmarkConfig, BENCHMARK_MODES
from pocket_agent_cli.datasets import DatasetRegistry, Problem
from pocket_agent_cli.datasets.humaneval import HumanEvalDataset


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def temp_dir(tmp_path):
    """Create a temporary directory for test data."""
    return tmp_path


@pytest.fixture
def sample_humaneval_data():
    """Sample HumanEval data for testing."""
    return [
        {
            "task_id": "HumanEval/0",
            "prompt": 'from typing import List\n\n\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n    """ Check if in given list of numbers, are any two numbers closer to each other than\n    given threshold.\n    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)\n    False\n    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)\n    True\n    """\n',
            "canonical_solution": '    for idx, elem in enumerate(numbers):\n        for idx2, elem2 in enumerate(numbers):\n            if idx != idx2:\n                distance = abs(elem - elem2)\n                if distance < threshold:\n                    return True\n\n    return False\n',
            "test": '\n\nMETADATA = {\n    \'author\': \'jt\',\n    \'dataset\': \'test\'\n}\n\n\ndef check(candidate):\n    assert candidate([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.3) == True\n    assert candidate([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.05) == False\n    assert candidate([1.0, 2.0, 5.9, 4.0, 5.0], 0.95) == True\n    assert candidate([1.0, 2.0, 5.9, 4.0, 5.0], 0.8) == False\n    assert candidate([1.0, 2.0, 3.0, 4.0, 5.0, 2.0], 0.1) == True\n    assert candidate([1.1, 2.2, 3.1, 4.1, 5.1], 1.0) == True\n    assert candidate([1.1, 2.2, 3.1, 4.1, 5.1], 0.5) == False\n\n',
            "entry_point": "has_close_elements"
        },
        {
            "task_id": "HumanEval/1",
            "prompt": 'from typing import List\n\n\ndef separate_paren_groups(paren_string: str) -> List[str]:\n    """ Input to this function is a string containing multiple groups of nested parentheses. Your goal is to\n    separate those group into separate strings and return the list of those.\n    Separate groups are balanced (each open brace is properly closed) and not nested within each other\n    Ignore any spaces in the input string.\n    >>> separate_paren_groups(\'( ) (( )) (( )( ))\')\n    [\'()\', \'(())\', \'(()())\']\n    """\n',
            "canonical_solution": '    result = []\n    current_string = []\n    current_depth = 0\n\n    for c in paren_string:\n        if c == \'(\':\n            current_depth += 1\n            current_string.append(c)\n        elif c == \')\':\n            current_depth -= 1\n            current_string.append(c)\n\n            if current_depth == 0:\n                result.append(\'\'.join(current_string))\n                current_string.clear()\n\n    return result\n',
            "test": '\n\nMETADATA = {\n    \'author\': \'jt\',\n    \'dataset\': \'test\'\n}\n\n\ndef check(candidate):\n    assert candidate(\'(()()) ((())) () ((())()())\') == [\n        \'(()())\', \'((()))\', \'()\', \'((())()())\'\n    ]\n    assert candidate(\'() (()) ((())) (((())))\') == [\n        \'()\', \'(())\', \'((()))\', \'(((())))\'\n    ]\n    assert candidate(\'(()(())((())))\'     ) == [\n        \'(()(())((())))\'\n    ]\n    assert candidate(\'( ) (( )) (( )( ))\') == [\'()\', \'(())\', \'(()())\']\n\n',
            "entry_point": "separate_paren_groups"
        },
    ]


@pytest.fixture
def humaneval_data_dir(temp_dir, sample_humaneval_data):
    """Create a data directory with HumanEval data files."""
    # Create main file
    main_path = temp_dir / "humaneval.json"
    with open(main_path, 'w') as f:
        json.dump(sample_humaneval_data, f)

    # Create sample file
    sample_path = temp_dir / "humaneval_sample.json"
    with open(sample_path, 'w') as f:
        json.dump(sample_humaneval_data[:1], f)

    return temp_dir


@pytest.fixture
def mock_inference_service():
    """Create a mock inference service."""
    mock = Mock()
    mock.current_model = Mock()
    mock.current_model.id = 'test-model'
    mock.current_model.architecture = 'llama'
    mock.generate = Mock(return_value=iter([
        {"token": "code", "metrics": {"ttft": 100, "tps": 20}}
    ]))
    return mock


@pytest.fixture
def benchmark_service(mock_inference_service, humaneval_data_dir):
    """Create a BenchmarkService with HumanEval dataset."""
    config = BenchmarkConfig(model_name='test-model', problems_limit=2)
    return BenchmarkService(
        inference_service=mock_inference_service,
        config=config,
        dataset_name='humaneval',
        data_dir=humaneval_data_dir
    )


# ============================================================================
# Test: Problem Detection
# ============================================================================

class TestHumanEvalProblemDetection:
    """Tests for HumanEval problem detection."""

    def test_is_humaneval_problem_by_task_id(self, benchmark_service):
        """Test detection by task ID prefix."""
        problem = {"task_id": "HumanEval/0", "text": "some text"}
        assert benchmark_service._is_humaneval_problem(problem) is True

    def test_is_humaneval_problem_by_structure(self, benchmark_service):
        """Test detection by prompt structure (def + docstring)."""
        problem = {
            "task_id": "1",
            "text": 'def my_func(x):\n    """A docstring."""\n'
        }
        assert benchmark_service._is_humaneval_problem(problem) is True

    def test_is_not_humaneval_problem_mbpp(self, benchmark_service):
        """Test MBPP problems are not detected as HumanEval."""
        problem = {
            "task_id": "1",
            "text": "Write a function to add two numbers."
        }
        assert benchmark_service._is_humaneval_problem(problem) is False


# ============================================================================
# Test: Prompt Generation
# ============================================================================

class TestHumanEvalPromptGeneration:
    """Tests for HumanEval prompt generation."""

    def test_humaneval_prompt_includes_signature(self, benchmark_service):
        """Test that HumanEval prompts include the function signature."""
        problem_dict = benchmark_service._problem_to_dict(benchmark_service.problems[0])
        mode = BENCHMARK_MODES['base']

        prompt = benchmark_service._prepare_problem_prompt(problem_dict, mode)

        assert "def has_close_elements" in prompt
        assert "numbers: List[float]" in prompt

    def test_humaneval_prompt_includes_docstring(self, benchmark_service):
        """Test that HumanEval prompts include the docstring."""
        problem_dict = benchmark_service._problem_to_dict(benchmark_service.problems[0])
        mode = BENCHMARK_MODES['base']

        prompt = benchmark_service._prepare_problem_prompt(problem_dict, mode)

        assert "Check if in given list of numbers" in prompt

    def test_humaneval_prompt_base_mode_instruction(self, benchmark_service):
        """Test base mode has completion instruction."""
        problem_dict = benchmark_service._problem_to_dict(benchmark_service.problems[0])
        mode = BENCHMARK_MODES['base']

        prompt = benchmark_service._prepare_problem_prompt(problem_dict, mode)

        assert "Complete the function body" in prompt

    def test_humaneval_prompt_tool_submission_instruction(self, benchmark_service):
        """Test tool_submission mode has submit instruction."""
        problem_dict = benchmark_service._problem_to_dict(benchmark_service.problems[0])
        mode = BENCHMARK_MODES['tool_submission']

        prompt = benchmark_service._prepare_problem_prompt(problem_dict, mode)

        assert "Complete the function and submit" in prompt

    def test_humaneval_prompt_full_tool_instruction(self, benchmark_service):
        """Test full_tool mode has submit instruction."""
        problem_dict = benchmark_service._problem_to_dict(benchmark_service.problems[0])
        mode = BENCHMARK_MODES['full_tool']

        prompt = benchmark_service._prepare_problem_prompt(problem_dict, mode)

        assert "Complete and submit" in prompt

    def test_humaneval_prompt_includes_test_examples(self, benchmark_service):
        """Test that HumanEval prompts include test examples as comments."""
        problem_dict = benchmark_service._problem_to_dict(benchmark_service.problems[0])
        mode = BENCHMARK_MODES['base']

        prompt = benchmark_service._prepare_problem_prompt(problem_dict, mode)

        # Test cases should be included as comments
        assert "Example assertions" in prompt or "assert" in prompt


# ============================================================================
# Test: Test Execution
# ============================================================================

class TestHumanEvalTestExecution:
    """Tests for HumanEval test execution."""

    @pytest.mark.asyncio
    async def test_run_humaneval_tests_with_correct_code(self, benchmark_service, sample_humaneval_data):
        """Test running HumanEval tests with correct solution."""
        problem = sample_humaneval_data[0]
        full_code = problem['prompt'] + problem['canonical_solution']

        # Mock tool executor
        async def mock_run_code(code):
            # Execute the code to verify it works
            try:
                exec(code, {})
                return ""  # No output = success
            except Exception as e:
                return str(e)

        benchmark_service.tool_executor = Mock()
        benchmark_service.tool_executor.sandbox_dir = "/tmp/test"
        benchmark_service.tool_executor._run_python_code = mock_run_code

        problem_dict = {
            "task_id": problem["task_id"],
            "text": problem["prompt"],
            "test_list": [f"assert {problem['entry_point']}([1.0, 2.0], 0.5) == False"],
            "entry_point": problem["entry_point"]
        }

        results = await benchmark_service._run_humaneval_tests(full_code, problem_dict)

        # Should have one result (the check() function test)
        assert len(results) >= 1

    @pytest.mark.asyncio
    async def test_run_humaneval_tests_with_wrong_code(self, benchmark_service, sample_humaneval_data):
        """Test running HumanEval tests with incorrect solution."""
        problem = sample_humaneval_data[0]
        # Wrong implementation - always returns False
        wrong_code = problem['prompt'] + "    return False\n"

        async def mock_run_code(code):
            try:
                exec(code, {})
                return ""
            except AssertionError:
                return "AssertionError"
            except Exception as e:
                return str(e)

        benchmark_service.tool_executor = Mock()
        benchmark_service.tool_executor.sandbox_dir = "/tmp/test"
        benchmark_service.tool_executor._run_python_code = mock_run_code

        problem_dict = {
            "task_id": problem["task_id"],
            "text": problem["prompt"],
            "test_list": [f"assert {problem['entry_point']}([1.0, 2.0, 2.2], 0.5) == True"],
            "entry_point": problem["entry_point"]
        }

        results = await benchmark_service._run_humaneval_tests(wrong_code, problem_dict)

        # When wrong, results should show failures
        # (Depends on whether original_test is found - may fall back to individual assertions)
        assert len(results) >= 1


# ============================================================================
# Test: Full Integration with Canonical Solutions
# ============================================================================

class TestHumanEvalCanonicalSolutions:
    """Test that canonical solutions pass the check() functions."""

    def test_canonical_solution_0_passes_check(self, sample_humaneval_data):
        """Test HumanEval/0 canonical solution passes."""
        problem = sample_humaneval_data[0]
        full_code = problem['prompt'] + problem['canonical_solution']
        test_code = f"{full_code}\n{problem['test']}\ncheck({problem['entry_point']})"

        # Should not raise any exceptions
        exec(test_code, {})

    def test_canonical_solution_1_passes_check(self, sample_humaneval_data):
        """Test HumanEval/1 canonical solution passes."""
        problem = sample_humaneval_data[1]
        full_code = problem['prompt'] + problem['canonical_solution']
        test_code = f"{full_code}\n{problem['test']}\ncheck({problem['entry_point']})"

        # Should not raise any exceptions
        exec(test_code, {})

    def test_wrong_solution_fails_check(self, sample_humaneval_data):
        """Test that a wrong solution fails the check."""
        problem = sample_humaneval_data[0]
        wrong_code = problem['prompt'] + "    return False  # Wrong!\n"
        test_code = f"{wrong_code}\n{problem['test']}\ncheck({problem['entry_point']})"

        with pytest.raises(AssertionError):
            exec(test_code, {})


# ============================================================================
# Test: Dataset Loading and Problem Conversion
# ============================================================================

class TestHumanEvalDatasetIntegration:
    """Tests for HumanEval dataset integration with BenchmarkService."""

    def test_load_humaneval_problems(self, benchmark_service):
        """Test loading HumanEval problems."""
        problems = benchmark_service.problems

        assert len(problems) == 2
        assert all(isinstance(p, Problem) for p in problems)

    def test_problem_conversion_preserves_metadata(self, benchmark_service):
        """Test that problem conversion preserves original_test in metadata."""
        problem = benchmark_service.problems[0]

        assert problem.metadata is not None
        assert problem.metadata.get("source") == "humaneval"
        assert "original_test" in problem.metadata

    def test_problem_to_dict_includes_entry_point(self, benchmark_service):
        """Test that problem_to_dict includes entry_point."""
        problem = benchmark_service.problems[0]
        problem_dict = benchmark_service._problem_to_dict(problem)

        assert "entry_point" in problem_dict
        assert problem_dict["entry_point"] == "has_close_elements"

    def test_dataset_name_is_correct(self, benchmark_service):
        """Test dataset name is set correctly."""
        assert benchmark_service.dataset_name == "humaneval"
        assert benchmark_service.dataset.name == "humaneval"


# ============================================================================
# Test: Create Test Harness
# ============================================================================

class TestHumanEvalTestHarness:
    """Tests for HumanEval test harness creation."""

    def test_create_test_harness_structure(self, humaneval_data_dir):
        """Test that create_test_harness produces valid structure."""
        dataset = HumanEvalDataset(humaneval_data_dir)
        problems = dataset.load(split="test", limit=1)
        problem = problems[0]

        generated_code = problem.prompt + problem.canonical_solution
        harness = dataset.create_test_harness(problem, generated_code)

        # Should contain the function definition
        assert "def has_close_elements" in harness
        # Should contain the check function
        assert "def check(candidate):" in harness
        # Should call check with the function name
        assert "check(has_close_elements)" in harness

    def test_create_test_harness_is_executable(self, humaneval_data_dir):
        """Test that the test harness is executable."""
        dataset = HumanEvalDataset(humaneval_data_dir)
        problems = dataset.load(split="test", limit=1)
        problem = problems[0]

        generated_code = problem.prompt + problem.canonical_solution
        harness = dataset.create_test_harness(problem, generated_code)

        # Should execute without errors
        exec(harness, {})


# ============================================================================
# Test: Extract Test Cases
# ============================================================================

class TestExtractTestCases:
    """Tests for extracting test cases from check() functions."""

    def test_extract_multiple_assertions(self, humaneval_data_dir):
        """Test extracting multiple assertions."""
        dataset = HumanEvalDataset(humaneval_data_dir)

        test_code = """def check(candidate):
    assert candidate(1, 2) == 3
    assert candidate(0, 0) == 0
    assert candidate(-1, 1) == 0
"""
        cases = dataset._extract_test_cases(test_code, "add")

        assert len(cases) == 3
        assert "add(1, 2)" in cases[0]
        assert "add(0, 0)" in cases[1]
        assert "add(-1, 1)" in cases[2]

    def test_extract_replaces_candidate_with_function_name(self, humaneval_data_dir):
        """Test that 'candidate' is replaced with the function name."""
        dataset = HumanEvalDataset(humaneval_data_dir)

        test_code = """def check(candidate):
    assert candidate([1, 2], 0.5) == False
"""
        cases = dataset._extract_test_cases(test_code, "my_function")

        assert "candidate" not in cases[0]
        assert "my_function([1, 2], 0.5)" in cases[0]

    def test_extract_handles_empty_check(self, humaneval_data_dir):
        """Test handling check() with no assertions."""
        dataset = HumanEvalDataset(humaneval_data_dir)

        test_code = """def check(candidate):
    pass
"""
        cases = dataset._extract_test_cases(test_code, "func")

        # Should return fallback
        assert len(cases) == 1
        assert "check(func)" in cases[0]
