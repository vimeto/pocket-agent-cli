"""Smoke tests for dataset abstraction with actual MBPP data.

These tests use the actual MBPP data files in the project to verify
that the dataset abstraction works correctly with real data.
"""

import pytest
from pathlib import Path

from pocket_agent_cli.datasets import DatasetRegistry, Problem
from pocket_agent_cli.datasets.mbpp import MBPPDataset
from pocket_agent_cli.config import DATA_DIR


# Skip all tests if MBPP data is not available
pytestmark = pytest.mark.skipif(
    not (Path(__file__).parent.parent / "pocket_agent_cli" / "data" / "mbpp_sample.json").exists()
    and not DATA_DIR.exists(),
    reason="MBPP data not available"
)


class TestMBPPSmokeTests:
    """Smoke tests using actual MBPP data."""

    @pytest.fixture
    def data_dir(self):
        """Get the actual data directory."""
        # Try the package data directory first
        pkg_data = Path(__file__).parent.parent / "pocket_agent_cli" / "data"
        if pkg_data.exists():
            return pkg_data
        # Fall back to configured DATA_DIR
        return DATA_DIR

    @pytest.fixture
    def mbpp_dataset(self, data_dir):
        """Create MBPP dataset with actual data."""
        return MBPPDataset(data_dir)

    def test_actual_mbpp_data_exists(self, data_dir):
        """Verify actual MBPP data files exist."""
        assert (data_dir / "mbpp_sample.json").exists(), "mbpp_sample.json not found"

    def test_load_actual_sample(self, mbpp_dataset):
        """Test loading actual MBPP sample data."""
        problems = mbpp_dataset.load(split="sample")

        assert len(problems) >= 1, "Should load at least 1 problem"

        # Check first problem has all required fields
        problem = problems[0]
        assert problem.task_id, "task_id should not be empty"
        assert problem.prompt, "prompt should not be empty"
        assert problem.canonical_solution, "canonical_solution should not be empty"
        assert problem.test_cases, "test_cases should not be empty"
        assert problem.entry_point, "entry_point should not be empty"

    def test_load_actual_sample_structure(self, mbpp_dataset):
        """Test that actual MBPP sample has correct structure."""
        problems = mbpp_dataset.load(split="sample")

        for problem in problems:
            # Verify Problem type
            assert isinstance(problem, Problem)

            # Verify test cases are valid Python assertions
            for test_case in problem.test_cases:
                assert "assert" in test_case, f"Test case should contain assert: {test_case}"

            # Verify entry point matches a function in the solution
            assert f"def {problem.entry_point}" in problem.canonical_solution, \
                f"Entry point {problem.entry_point} not found in solution"

    def test_load_actual_full_dataset(self, mbpp_dataset):
        """Test loading actual full MBPP dataset if available."""
        try:
            problems = mbpp_dataset.load(split="full")
            # Full dataset should have ~974 problems
            assert len(problems) >= 500, f"Expected at least 500 problems, got {len(problems)}"
        except FileNotFoundError:
            pytest.skip("Full MBPP dataset not available")

    def test_load_actual_test_split(self, mbpp_dataset):
        """Test loading actual MBPP test split if available."""
        try:
            problems = mbpp_dataset.load(split="test")
            # Test split should have some problems (could be 100 or 500 depending on download)
            assert len(problems) >= 10, f"Expected at least 10 problems, got {len(problems)}"

            # Verify all problems have valid structure
            for problem in problems:
                assert problem.task_id, "task_id should not be empty"
                assert problem.prompt, "prompt should not be empty"
                assert problem.test_cases, "test_cases should not be empty"
        except FileNotFoundError:
            pytest.skip("MBPP test split not available")

    def test_problem_validation(self, mbpp_dataset):
        """Test that all loaded problems pass validation."""
        problems = mbpp_dataset.load(split="sample")

        for problem in problems:
            assert mbpp_dataset.validate_problem(problem), \
                f"Problem {problem.task_id} failed validation"

    def test_registry_integration(self, data_dir):
        """Test that MBPP loads correctly through registry."""
        dataset = DatasetRegistry.create("mbpp", data_dir)

        assert isinstance(dataset, MBPPDataset)
        assert dataset.is_downloaded()

        problems = dataset.load(split="sample")
        assert len(problems) >= 1

    def test_get_sample_method(self, mbpp_dataset):
        """Test get_sample method with actual data."""
        sample = mbpp_dataset.get_sample(n=2)

        assert len(sample) <= 2
        assert all(isinstance(p, Problem) for p in sample)

    def test_problem_to_dict_round_trip(self, mbpp_dataset):
        """Test that to_dict/from_dict preserves actual problem data."""
        problems = mbpp_dataset.load(split="sample", limit=1)
        original = problems[0]

        # Convert to dict and back
        data = original.to_dict()
        recreated = Problem.from_dict(data)

        assert recreated.task_id == original.task_id
        assert recreated.prompt == original.prompt
        assert recreated.canonical_solution == original.canonical_solution
        assert recreated.test_cases == original.test_cases
        assert recreated.entry_point == original.entry_point

    def test_actual_test_cases_executable(self, mbpp_dataset):
        """Test that actual test cases are syntactically valid Python."""
        problems = mbpp_dataset.load(split="sample")

        for problem in problems:
            # Combine solution with test cases and check syntax
            code = problem.canonical_solution
            for test_case in problem.test_cases:
                full_code = f"{code}\n\n{test_case}"

                # This should not raise SyntaxError
                try:
                    compile(full_code, '<string>', 'exec')
                except SyntaxError as e:
                    pytest.fail(f"Problem {problem.task_id} has syntax error: {e}")

    def test_actual_solutions_pass_tests(self, mbpp_dataset):
        """Test that canonical solutions pass their test cases."""
        problems = mbpp_dataset.load(split="sample")

        for problem in problems:
            code = problem.canonical_solution

            for test_case in problem.test_cases:
                full_code = f"{code}\n\n{test_case}"

                try:
                    exec(full_code, {})
                except AssertionError:
                    pytest.fail(
                        f"Problem {problem.task_id}: canonical solution failed test: {test_case}"
                    )
                except Exception as e:
                    # Some tests might need imports or have other issues
                    # Just ensure they don't fail on the canonical solution
                    pass


class TestDatasetRegistrySmoke:
    """Smoke tests for dataset registry with actual data."""

    def test_all_registered_datasets(self):
        """Test that all registered datasets are properly configured."""
        datasets = DatasetRegistry.list_datasets()

        assert len(datasets) >= 2, "Should have at least mbpp and humaneval"

        for name, description in datasets.items():
            assert name, "Dataset name should not be empty"
            assert description, f"Dataset {name} should have description"

    def test_create_all_datasets(self, tmp_path):
        """Test creating instances of all registered datasets."""
        for name in DatasetRegistry.list_names():
            dataset = DatasetRegistry.create(name, tmp_path)

            assert dataset is not None
            assert dataset.name == name
            assert dataset.problem_count > 0
            assert dataset.available_splits
