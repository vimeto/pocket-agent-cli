"""Comprehensive tests for the dataset abstraction layer."""

import pytest
import json
from pathlib import Path
from unittest.mock import patch, MagicMock

from pocket_agent_cli.datasets import Dataset, Problem, DatasetRegistry
from pocket_agent_cli.datasets.mbpp import MBPPDataset
from pocket_agent_cli.datasets.humaneval import HumanEvalDataset


# ============================================================================
# Problem Dataclass Tests
# ============================================================================

class TestProblem:
    """Tests for the Problem dataclass."""

    def test_problem_creation(self):
        """Test creating a Problem with all fields."""
        problem = Problem(
            task_id="1",
            prompt="Write a function to add two numbers.",
            canonical_solution="def add(a, b):\n    return a + b",
            test_cases=["assert add(1, 2) == 3"],
            entry_point="add",
        )

        assert problem.task_id == "1"
        assert problem.prompt == "Write a function to add two numbers."
        assert problem.canonical_solution == "def add(a, b):\n    return a + b"
        assert problem.test_cases == ["assert add(1, 2) == 3"]
        assert problem.entry_point == "add"
        assert problem.metadata == {}

    def test_problem_with_metadata(self):
        """Test creating a Problem with metadata."""
        problem = Problem(
            task_id="1",
            prompt="Test",
            canonical_solution="code",
            test_cases=["test"],
            entry_point="func",
            metadata={"source": "mbpp", "difficulty": "easy"},
        )

        assert problem.metadata == {"source": "mbpp", "difficulty": "easy"}

    def test_problem_metadata_defaults_to_empty_dict(self):
        """Test that metadata defaults to empty dict."""
        problem = Problem(
            task_id="1",
            prompt="Test",
            canonical_solution="code",
            test_cases=["test"],
            entry_point="func",
            metadata=None,
        )

        assert problem.metadata == {}

    def test_problem_to_dict(self):
        """Test converting Problem to dictionary."""
        problem = Problem(
            task_id="42",
            prompt="Test prompt",
            canonical_solution="def test(): pass",
            test_cases=["assert test() is None"],
            entry_point="test",
            metadata={"key": "value"},
        )

        data = problem.to_dict()

        assert data["task_id"] == "42"
        assert data["prompt"] == "Test prompt"
        assert data["canonical_solution"] == "def test(): pass"
        assert data["test_cases"] == ["assert test() is None"]
        assert data["entry_point"] == "test"
        assert data["metadata"] == {"key": "value"}

    def test_problem_from_dict(self):
        """Test creating Problem from dictionary."""
        data = {
            "task_id": 123,  # Should be converted to string
            "prompt": "Test",
            "canonical_solution": "code",
            "test_cases": ["test1", "test2"],
            "entry_point": "func",
            "metadata": {"source": "test"},
        }

        problem = Problem.from_dict(data)

        assert problem.task_id == "123"
        assert problem.prompt == "Test"
        assert problem.test_cases == ["test1", "test2"]
        assert problem.metadata == {"source": "test"}

    def test_problem_round_trip(self):
        """Test that to_dict and from_dict are inverse operations."""
        original = Problem(
            task_id="99",
            prompt="Original prompt",
            canonical_solution="def original(): return 99",
            test_cases=["assert original() == 99"],
            entry_point="original",
            metadata={"round": "trip"},
        )

        recreated = Problem.from_dict(original.to_dict())

        assert recreated.task_id == original.task_id
        assert recreated.prompt == original.prompt
        assert recreated.canonical_solution == original.canonical_solution
        assert recreated.test_cases == original.test_cases
        assert recreated.entry_point == original.entry_point
        assert recreated.metadata == original.metadata


# ============================================================================
# DatasetRegistry Tests
# ============================================================================

class TestDatasetRegistry:
    """Tests for the DatasetRegistry."""

    def test_mbpp_is_registered(self):
        """Test that MBPP dataset is automatically registered."""
        assert DatasetRegistry.is_registered("mbpp")

    def test_humaneval_is_registered(self):
        """Test that HumanEval dataset is automatically registered."""
        assert DatasetRegistry.is_registered("humaneval")

    def test_list_datasets(self):
        """Test listing all registered datasets."""
        datasets = DatasetRegistry.list_datasets()

        assert "mbpp" in datasets
        assert "humaneval" in datasets
        assert isinstance(datasets["mbpp"], str)
        assert isinstance(datasets["humaneval"], str)

    def test_list_names(self):
        """Test listing dataset names."""
        names = DatasetRegistry.list_names()

        assert "mbpp" in names
        assert "humaneval" in names

    def test_get_registered_dataset(self):
        """Test getting a registered dataset class."""
        mbpp_cls = DatasetRegistry.get("mbpp")

        assert mbpp_cls is MBPPDataset

    def test_get_unregistered_dataset(self):
        """Test getting an unregistered dataset returns None."""
        result = DatasetRegistry.get("nonexistent")

        assert result is None

    def test_create_dataset(self, temp_dir):
        """Test creating a dataset instance."""
        dataset = DatasetRegistry.create("mbpp", temp_dir)

        assert isinstance(dataset, MBPPDataset)
        assert dataset.data_dir == temp_dir

    def test_create_unknown_dataset(self, temp_dir):
        """Test creating unknown dataset raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            DatasetRegistry.create("unknown_dataset", temp_dir)

        assert "Unknown dataset" in str(exc_info.value)
        assert "unknown_dataset" in str(exc_info.value)

    def test_count(self):
        """Test counting registered datasets."""
        count = DatasetRegistry.count()

        assert count >= 2  # At least mbpp and humaneval


# ============================================================================
# MBPPDataset Tests
# ============================================================================

class TestMBPPDataset:
    """Tests for MBPPDataset."""

    @pytest.fixture
    def mbpp_dataset(self, temp_dir):
        """Create MBPP dataset instance."""
        return MBPPDataset(temp_dir)

    @pytest.fixture
    def sample_mbpp_data(self):
        """Sample MBPP data for testing."""
        return [
            {
                "task_id": 1,
                "text": "Write a function to find the minimum cost path.",
                "code": "def min_cost(cost, m, n):\n    return cost[m][n]",
                "test_list": [
                    "assert min_cost([[1, 2], [3, 4]], 1, 1) == 4",
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
                "text": "Write a function to multiply two numbers.",
                "code": "def multiply(a, b):\n    return a * b",
                "test_list": [
                    "assert multiply(2, 3) == 6",
                ]
            },
        ]

    @pytest.fixture
    def mbpp_with_data(self, mbpp_dataset, sample_mbpp_data):
        """MBPP dataset with sample data files created."""
        # Create sample file
        sample_path = mbpp_dataset.data_dir / "mbpp_sample.json"
        with open(sample_path, 'w') as f:
            json.dump(sample_mbpp_data, f)

        # Create full file
        full_path = mbpp_dataset.data_dir / "mbpp_full.json"
        with open(full_path, 'w') as f:
            json.dump(sample_mbpp_data, f)

        return mbpp_dataset

    def test_mbpp_properties(self, mbpp_dataset):
        """Test MBPP dataset properties."""
        assert mbpp_dataset.name == "mbpp"
        assert mbpp_dataset.problem_count == 974
        assert "full" in mbpp_dataset.available_splits
        assert "test" in mbpp_dataset.available_splits
        assert "sample" in mbpp_dataset.available_splits

    def test_is_downloaded_false(self, mbpp_dataset):
        """Test is_downloaded returns False when no files exist."""
        assert mbpp_dataset.is_downloaded() is False

    def test_is_downloaded_true(self, mbpp_with_data):
        """Test is_downloaded returns True when files exist."""
        assert mbpp_with_data.is_downloaded() is True

    def test_load_sample(self, mbpp_with_data):
        """Test loading sample split."""
        problems = mbpp_with_data.load(split="sample")

        assert len(problems) == 3
        assert all(isinstance(p, Problem) for p in problems)

    def test_load_with_limit(self, mbpp_with_data):
        """Test loading with limit."""
        problems = mbpp_with_data.load(split="sample", limit=2)

        assert len(problems) == 2

    def test_load_converts_to_problem(self, mbpp_with_data):
        """Test that load converts to Problem objects correctly."""
        problems = mbpp_with_data.load(split="sample", limit=1)

        problem = problems[0]
        assert problem.task_id == "1"
        assert "minimum cost path" in problem.prompt
        assert problem.entry_point == "min_cost"
        assert len(problem.test_cases) == 1
        assert problem.metadata["source"] == "mbpp"

    def test_load_invalid_split(self, mbpp_with_data):
        """Test loading invalid split raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            mbpp_with_data.load(split="invalid")

        assert "Invalid split" in str(exc_info.value)

    def test_load_not_downloaded(self, mbpp_dataset):
        """Test loading when not downloaded raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            mbpp_dataset.load(split="test")

    def test_extract_function_name(self, mbpp_dataset):
        """Test function name extraction."""
        code1 = "def my_function(x, y):\n    return x + y"
        assert mbpp_dataset._extract_function_name(code1) == "my_function"

        code2 = "def another_func():\n    pass"
        assert mbpp_dataset._extract_function_name(code2) == "another_func"

        code3 = "# No function here"
        assert mbpp_dataset._extract_function_name(code3) == "solution"

    def test_get_sample(self, mbpp_with_data):
        """Test get_sample method."""
        sample = mbpp_with_data.get_sample(n=2)

        assert len(sample) == 2
        assert all(isinstance(p, Problem) for p in sample)

    def test_validate_problem(self, mbpp_dataset):
        """Test problem validation."""
        valid_problem = Problem(
            task_id="1",
            prompt="Test",
            canonical_solution="code",
            test_cases=["test"],
            entry_point="func",
        )
        assert mbpp_dataset.validate_problem(valid_problem) is True

        invalid_problem = Problem(
            task_id="",  # Empty task_id
            prompt="Test",
            canonical_solution="code",
            test_cases=["test"],
            entry_point="func",
        )
        assert mbpp_dataset.validate_problem(invalid_problem) is False

    def test_repr(self, mbpp_dataset):
        """Test string representation."""
        repr_str = repr(mbpp_dataset)

        assert "MBPPDataset" in repr_str
        assert "mbpp" in repr_str

    @patch('requests.get')
    def test_download_success(self, mock_get, mbpp_dataset):
        """Test successful download."""
        # Mock response with JSONL data
        mock_response = MagicMock()
        mock_response.text = '{"task_id": 1, "text": "Test", "code": "def f(): pass", "test_list": ["assert True"]}\n'
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        result = mbpp_dataset.download()

        assert result is True
        assert (mbpp_dataset.data_dir / "mbpp_full.json").exists()
        assert (mbpp_dataset.data_dir / "mbpp_sample.json").exists()

    @patch('requests.get')
    def test_download_failure(self, mock_get, mbpp_dataset):
        """Test download failure handling."""
        import requests
        mock_get.side_effect = requests.RequestException("Network error")

        result = mbpp_dataset.download()

        assert result is False


# ============================================================================
# HumanEvalDataset Tests
# ============================================================================

class TestHumanEvalDataset:
    """Tests for HumanEvalDataset."""

    @pytest.fixture
    def humaneval_dataset(self, temp_dir):
        """Create HumanEval dataset instance."""
        return HumanEvalDataset(temp_dir)

    @pytest.fixture
    def sample_humaneval_data(self):
        """Sample HumanEval data for testing."""
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
            },
        ]

    @pytest.fixture
    def humaneval_with_data(self, humaneval_dataset, sample_humaneval_data):
        """HumanEval dataset with sample data files created."""
        # Create main file
        main_path = humaneval_dataset.data_dir / "humaneval.json"
        with open(main_path, 'w') as f:
            json.dump(sample_humaneval_data, f)

        # Create sample file
        sample_path = humaneval_dataset.data_dir / "humaneval_sample.json"
        with open(sample_path, 'w') as f:
            json.dump(sample_humaneval_data[:1], f)

        return humaneval_dataset

    def test_humaneval_properties(self, humaneval_dataset):
        """Test HumanEval dataset properties."""
        assert humaneval_dataset.name == "humaneval"
        assert humaneval_dataset.problem_count == 164
        assert "test" in humaneval_dataset.available_splits
        assert "sample" in humaneval_dataset.available_splits

    def test_is_downloaded_false(self, humaneval_dataset):
        """Test is_downloaded returns False when no files exist."""
        assert humaneval_dataset.is_downloaded() is False

    def test_is_downloaded_true(self, humaneval_with_data):
        """Test is_downloaded returns True when files exist."""
        assert humaneval_with_data.is_downloaded() is True

    def test_load_test_split(self, humaneval_with_data):
        """Test loading test split."""
        problems = humaneval_with_data.load(split="test")

        assert len(problems) == 2
        assert all(isinstance(p, Problem) for p in problems)

    def test_load_sample_split(self, humaneval_with_data):
        """Test loading sample split."""
        problems = humaneval_with_data.load(split="sample")

        assert len(problems) == 1

    def test_load_with_limit(self, humaneval_with_data):
        """Test loading with limit."""
        problems = humaneval_with_data.load(split="test", limit=1)

        assert len(problems) == 1

    def test_load_converts_to_problem(self, humaneval_with_data):
        """Test that load converts to Problem objects correctly."""
        problems = humaneval_with_data.load(split="test", limit=1)

        problem = problems[0]
        assert problem.task_id == "HumanEval/0"
        assert "has_close_elements" in problem.prompt
        assert problem.entry_point == "has_close_elements"
        assert len(problem.test_cases) >= 1
        assert problem.metadata["source"] == "humaneval"
        assert "original_test" in problem.metadata

    def test_load_invalid_split(self, humaneval_with_data):
        """Test loading invalid split raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            humaneval_with_data.load(split="invalid")

        assert "Invalid split" in str(exc_info.value)

    def test_load_not_downloaded(self, humaneval_dataset):
        """Test loading when not downloaded raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            humaneval_dataset.load(split="test")

    def test_extract_test_cases(self, humaneval_dataset):
        """Test extracting test cases from check() function."""
        test_code = """def check(candidate):
    assert candidate([1, 2], 0.5) == False
    assert candidate([1.0, 1.1], 0.5) == True
"""
        cases = humaneval_dataset._extract_test_cases(test_code, "my_func")

        assert len(cases) == 2
        assert "my_func" in cases[0]
        assert "candidate" not in cases[0]

    def test_extract_test_cases_empty(self, humaneval_dataset):
        """Test extracting from test code without assertions."""
        test_code = "def check(candidate):\n    pass"
        cases = humaneval_dataset._extract_test_cases(test_code, "func")

        assert len(cases) == 1
        assert "check(func)" in cases[0]

    def test_create_test_harness(self, humaneval_with_data):
        """Test creating test harness."""
        problems = humaneval_with_data.load(split="test", limit=1)
        problem = problems[0]

        generated_code = "def has_close_elements(numbers, threshold):\n    return False"
        harness = humaneval_with_data.create_test_harness(problem, generated_code)

        assert "def has_close_elements" in harness
        assert "def check(candidate):" in harness
        assert "check(has_close_elements)" in harness

    @patch('requests.get')
    def test_download_success(self, mock_get, humaneval_dataset):
        """Test successful download."""
        import gzip

        # Create mock gzipped JSONL data
        jsonl_data = '{"task_id": "HumanEval/0", "prompt": "def f():", "canonical_solution": "pass", "test": "def check(c): assert True", "entry_point": "f"}\n'
        gzipped_data = gzip.compress(jsonl_data.encode('utf-8'))

        mock_response = MagicMock()
        mock_response.content = gzipped_data
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        result = humaneval_dataset.download()

        assert result is True
        assert (humaneval_dataset.data_dir / "humaneval.json").exists()
        assert (humaneval_dataset.data_dir / "humaneval_sample.json").exists()

    @patch('requests.get')
    def test_download_failure(self, mock_get, humaneval_dataset):
        """Test download failure handling."""
        import requests
        mock_get.side_effect = requests.RequestException("Network error")

        result = humaneval_dataset.download()

        assert result is False


# ============================================================================
# Integration Tests
# ============================================================================

class TestDatasetIntegration:
    """Integration tests for the dataset system."""

    def test_create_and_load_mbpp(self, temp_dir, sample_problems):
        """Test creating MBPP dataset and loading problems."""
        # Create dataset through registry
        dataset = DatasetRegistry.create("mbpp", temp_dir)

        # Create sample data file
        sample_path = temp_dir / "mbpp_sample.json"
        with open(sample_path, 'w') as f:
            json.dump(sample_problems, f)

        # Load and verify
        problems = dataset.load(split="sample")

        assert len(problems) == 3
        assert problems[0].task_id == "1"
        assert problems[0].entry_point == "min_cost"

    def test_create_and_load_humaneval(self, temp_dir, sample_humaneval_problems):
        """Test creating HumanEval dataset and loading problems."""
        # Create dataset through registry
        dataset = DatasetRegistry.create("humaneval", temp_dir)

        # Create data file
        data_path = temp_dir / "humaneval.json"
        with open(data_path, 'w') as f:
            json.dump(sample_humaneval_problems, f)

        # Load and verify
        problems = dataset.load(split="test")

        assert len(problems) == 2
        assert problems[0].task_id == "HumanEval/0"
        assert problems[0].entry_point == "has_close_elements"

    def test_problem_compatibility(self, temp_dir, sample_problems, sample_humaneval_problems):
        """Test that problems from both datasets have compatible structure."""
        # Create MBPP data
        mbpp_path = temp_dir / "mbpp_sample.json"
        with open(mbpp_path, 'w') as f:
            json.dump(sample_problems, f)

        # Create HumanEval data
        humaneval_path = temp_dir / "humaneval.json"
        with open(humaneval_path, 'w') as f:
            json.dump(sample_humaneval_problems, f)

        # Load from both
        mbpp = DatasetRegistry.create("mbpp", temp_dir)
        humaneval = DatasetRegistry.create("humaneval", temp_dir)

        mbpp_problems = mbpp.load(split="sample", limit=1)
        humaneval_problems = humaneval.load(split="test", limit=1)

        # Both should have the same attributes
        for problem in mbpp_problems + humaneval_problems:
            assert hasattr(problem, "task_id")
            assert hasattr(problem, "prompt")
            assert hasattr(problem, "canonical_solution")
            assert hasattr(problem, "test_cases")
            assert hasattr(problem, "entry_point")
            assert hasattr(problem, "metadata")

            # All should be valid
            assert problem.task_id
            assert problem.prompt
            assert problem.entry_point
            assert problem.test_cases
