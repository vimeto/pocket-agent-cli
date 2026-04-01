"""Tests for the GSM8K dataset implementation."""

import pytest
import json
from pathlib import Path
from unittest.mock import patch, MagicMock

from pocket_agent_cli.datasets import Dataset, Problem, DatasetRegistry
from pocket_agent_cli.datasets.gsm8k import (
    GSM8KDataset,
    extract_numeric_answer,
    numeric_answers_match,
)


# ============================================================================
# Answer Extraction Tests
# ============================================================================


class TestExtractNumericAnswer:
    """Tests for the extract_numeric_answer function."""

    def test_gsm8k_format(self):
        """Test extraction from GSM8K #### format."""
        assert extract_numeric_answer("#### 42") == 42.0
        assert extract_numeric_answer("Some reasoning\n#### 100") == 100.0
        assert extract_numeric_answer("####42") == 42.0

    def test_the_answer_is(self):
        """Test extraction from 'The answer is X' format."""
        assert extract_numeric_answer("The answer is 42") == 42.0
        assert extract_numeric_answer("The answer is 42.5") == 42.5
        assert extract_numeric_answer("the answer is 7") == 7.0
        assert extract_numeric_answer("The final answer is 99") == 99.0
        assert extract_numeric_answer("The answer is: 55") == 55.0

    def test_equals_format(self):
        """Test extraction from '= X' format."""
        assert extract_numeric_answer("Total = 42") == 42.0
        assert extract_numeric_answer("x = 100") == 100.0

    def test_bare_number(self):
        """Test extraction of a bare number."""
        assert extract_numeric_answer("42") == 42.0
        assert extract_numeric_answer("  42  ") == 42.0
        assert extract_numeric_answer("3.14") == 3.14

    def test_negative_numbers(self):
        """Test extraction of negative numbers."""
        assert extract_numeric_answer("#### -42") == -42.0
        assert extract_numeric_answer("The answer is -7") == -7.0
        assert extract_numeric_answer("-100") == -100.0

    def test_comma_separated_numbers(self):
        """Test extraction of comma-separated large numbers."""
        assert extract_numeric_answer("#### 1,234") == 1234.0
        assert extract_numeric_answer("The answer is 1,234,567") == 1234567.0
        assert extract_numeric_answer("1,000") == 1000.0

    def test_decimal_numbers(self):
        """Test extraction of decimal numbers."""
        assert extract_numeric_answer("#### 3.14") == 3.14
        assert extract_numeric_answer("The answer is 0.5") == 0.5

    def test_dollar_amounts(self):
        """Test extraction of dollar amounts."""
        assert extract_numeric_answer("The answer is $42") == 42.0
        assert extract_numeric_answer("$1,234") == 1234.0

    def test_integer_vs_float(self):
        """Test that 42.0 and 42 are treated as equivalent."""
        assert extract_numeric_answer("42.0") == 42.0
        assert extract_numeric_answer("42") == 42.0

    def test_empty_and_none(self):
        """Test edge cases with empty/None input."""
        assert extract_numeric_answer("") is None
        assert extract_numeric_answer("   ") is None
        assert extract_numeric_answer(None) is None

    def test_no_number_found(self):
        """Test when no number is present."""
        assert extract_numeric_answer("no numbers here") is None

    def test_last_number_in_text(self):
        """Test that last number in longer text is extracted as fallback."""
        text = "First we get 10, then add 5, so the total is 15"
        assert extract_numeric_answer(text) == 15.0

    def test_multiline_text(self):
        """Test extraction from multiline text."""
        text = "Step 1: 10 + 20 = 30\nStep 2: 30 + 12 = 42\n#### 42"
        assert extract_numeric_answer(text) == 42.0

    def test_gsm8k_format_takes_priority(self):
        """Test that #### format is preferred over other patterns."""
        text = "The answer is 99\n#### 42"
        assert extract_numeric_answer(text) == 42.0


# ============================================================================
# Numeric Comparison Tests
# ============================================================================


class TestNumericAnswersMatch:
    """Tests for the numeric_answers_match function."""

    def test_exact_integer_match(self):
        """Test exact integer comparison."""
        assert numeric_answers_match(42.0, 42.0) is True
        assert numeric_answers_match(0.0, 0.0) is True
        assert numeric_answers_match(-5.0, -5.0) is True

    def test_integer_float_equivalence(self):
        """Test that 42.0 == 42 (integer stored as float)."""
        assert numeric_answers_match(42.0, 42) is True
        assert numeric_answers_match(42, 42.0) is True

    def test_different_integers(self):
        """Test that different integers don't match."""
        assert numeric_answers_match(42.0, 43.0) is False

    def test_float_tolerance(self):
        """Test floating point comparison with tolerance."""
        assert numeric_answers_match(3.14, 3.14) is True
        assert numeric_answers_match(3.14, 3.1400001) is True  # Within tolerance
        assert numeric_answers_match(3.14, 3.15) is False  # Outside tolerance

    def test_large_numbers(self):
        """Test comparison of large numbers."""
        assert numeric_answers_match(1234567.0, 1234567.0) is True
        assert numeric_answers_match(1234567.0, 1234568.0) is False


# ============================================================================
# GSM8K Dataset Tests
# ============================================================================


class TestGSM8KDataset:
    """Tests for GSM8KDataset."""

    @pytest.fixture
    def gsm8k_dataset(self, temp_dir):
        """Create GSM8K dataset instance."""
        return GSM8KDataset(temp_dir)

    @pytest.fixture
    def sample_gsm8k_data(self):
        """Sample GSM8K data for testing."""
        return [
            {
                "question": "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells every duck egg at the farmers' market for $2. How much does she make every day at the farmers' market?",
                "answer": "Janet sells 16 - 3 - 4 = <<16-3-4=9>>9 duck eggs a day.\nShe makes 9 * 2 = <<9*2=18>>$18 every day at the farmer's market.\n#### 18",
            },
            {
                "question": "A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total does it take?",
                "answer": "It takes 2/2=<<2/2=1>>1 bolt of white fiber\nSo the total bolts needed is 2+1=<<2+1=3>>3\n#### 3",
            },
            {
                "question": "Josh decides to try flipping a house. He buys a house for $80,000 and puts $50,000 in repairs. This increased the value of the house by 150%. How much profit did he make?",
                "answer": "The cost of the house and repairs came out to 80,000+50,000=$<<80000+50000=130000>>130,000\nHe increased the value of the house by 80,000*150%=$<<80000*1.5=120000>>120,000\nSo the new value of the house is 120,000+80,000=$<<120000+80000=200000>>200,000\nSo he made a profit of 200,000-130,000=$<<200000-130000=70000>>70,000\n#### 70000",
            },
            {
                "question": "Every day, Wendi feeds each of her chickens three cups of mixed chicken feed, containing seeds, mealworms and vegetables to help keep them healthy. She gives the chickens their feed in three separate meals. In the morning, she gives her flock of chickens 15 cups of feed. In the afternoon, she gives her chickens another 25 cups of feed. How many cups of feed does she need to give her chickens in the final meal of the day if the size of Wendi's flock is 20 chickens?",
                "answer": "If each chicken eats 3 cups of feed per day, then for 20 chickens they would need 3*20=<<3*20=60>>60 cups of feed per day.\nIf she feeds them 15 cups in the morning and 25 cups in the afternoon, then the amount of feed remaining for the final meal is 60-15-25=<<60-15-25=20>>20 cups.\n#### 20",
            },
            {
                "question": "Kylar went to the store to get water and snacks for their road trip. He bought 3 bottles of water for $1.50 each and 5 packets of snacks for $3 each. How much did he spend in total?",
                "answer": "The cost of 3 bottles of water is 3 * $1.50 = $<<3*1.5=4.5>>4.50.\nThe cost of 5 packets of snacks is 5 * $3 = $<<5*3=15>>15.\nKylar spent a total of $4.50 + $15 = $<<4.5+15=19.5>>19.50.\n#### 19.5",
            },
        ]

    @pytest.fixture
    def gsm8k_with_data(self, gsm8k_dataset, sample_gsm8k_data):
        """GSM8K dataset with sample data files created."""
        # Create test file
        test_path = gsm8k_dataset.data_dir / "gsm8k_test.json"
        with open(test_path, 'w') as f:
            json.dump(sample_gsm8k_data, f)

        # Create sample file
        sample_path = gsm8k_dataset.data_dir / "gsm8k_sample.json"
        with open(sample_path, 'w') as f:
            json.dump(sample_gsm8k_data[:3], f)

        return gsm8k_dataset

    # ---- Property tests ----

    def test_gsm8k_properties(self, gsm8k_dataset):
        """Test GSM8K dataset properties."""
        assert gsm8k_dataset.name == "gsm8k"
        assert gsm8k_dataset.problem_count == 1319
        assert "test" in gsm8k_dataset.available_splits
        assert "train" in gsm8k_dataset.available_splits
        assert "sample" in gsm8k_dataset.available_splits

    def test_gsm8k_description(self, gsm8k_dataset):
        """Test that description is set."""
        assert "grade school math" in gsm8k_dataset.description.lower()

    # ---- Download detection ----

    def test_is_downloaded_false(self, gsm8k_dataset):
        """Test is_downloaded returns False when no files exist."""
        assert gsm8k_dataset.is_downloaded() is False

    def test_is_downloaded_true(self, gsm8k_with_data):
        """Test is_downloaded returns True when files exist."""
        assert gsm8k_with_data.is_downloaded() is True

    # ---- Loading ----

    def test_load_test_split(self, gsm8k_with_data):
        """Test loading test split."""
        problems = gsm8k_with_data.load(split="test")

        assert len(problems) == 5
        assert all(isinstance(p, Problem) for p in problems)

    def test_load_sample_split(self, gsm8k_with_data):
        """Test loading sample split."""
        problems = gsm8k_with_data.load(split="sample")

        assert len(problems) == 3

    def test_load_with_limit(self, gsm8k_with_data):
        """Test loading with limit (subset selection)."""
        problems = gsm8k_with_data.load(split="test", limit=2)

        assert len(problems) == 2

    def test_load_converts_to_problem(self, gsm8k_with_data):
        """Test that load converts to Problem objects correctly."""
        problems = gsm8k_with_data.load(split="test", limit=1)

        problem = problems[0]
        assert problem.task_id == "GSM8K/0"
        assert "Janet" in problem.prompt
        assert "ducks" in problem.prompt
        assert problem.entry_point == "solve"
        assert len(problem.test_cases) == 1
        assert problem.metadata["source"] == "gsm8k"
        assert problem.metadata["ground_truth_answer"] == 18.0
        assert "#### 18" in problem.metadata["full_solution"]

    def test_load_invalid_split(self, gsm8k_with_data):
        """Test loading invalid split raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            gsm8k_with_data.load(split="invalid")

        assert "Invalid split" in str(exc_info.value)

    def test_load_not_downloaded(self, gsm8k_dataset):
        """Test loading when not downloaded raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            gsm8k_dataset.load(split="test")

    # ---- Ground truth extraction ----

    def test_extract_ground_truth(self):
        """Test ground truth extraction from GSM8K answer format."""
        assert GSM8KDataset._extract_ground_truth("some text\n#### 42") == 42.0
        assert GSM8KDataset._extract_ground_truth("#### 1,234") == 1234.0
        assert GSM8KDataset._extract_ground_truth("#### 3.14") == 3.14

    def test_extract_ground_truth_invalid(self):
        """Test ground truth extraction from invalid format raises ValueError."""
        with pytest.raises(ValueError):
            GSM8KDataset._extract_ground_truth("no answer here")

    # ---- Problem formatting ----

    def test_problem_has_correct_metadata(self, gsm8k_with_data):
        """Test that all problems have expected metadata fields."""
        problems = gsm8k_with_data.load(split="test")

        for problem in problems:
            assert "source" in problem.metadata
            assert problem.metadata["source"] == "gsm8k"
            assert "ground_truth_answer" in problem.metadata
            assert isinstance(problem.metadata["ground_truth_answer"], float)
            assert "full_solution" in problem.metadata

    def test_problem_task_ids_sequential(self, gsm8k_with_data):
        """Test that task IDs are sequential."""
        problems = gsm8k_with_data.load(split="test")

        for i, problem in enumerate(problems):
            assert problem.task_id == f"GSM8K/{i}"

    def test_problem_ground_truth_values(self, gsm8k_with_data):
        """Test ground truth extraction for all sample problems."""
        problems = gsm8k_with_data.load(split="test")

        expected_answers = [18.0, 3.0, 70000.0, 20.0, 19.5]
        for problem, expected in zip(problems, expected_answers):
            assert problem.metadata["ground_truth_answer"] == expected

    # ---- Evaluation ----

    def test_evaluate_response_correct(self, gsm8k_with_data):
        """Test evaluation of correct model responses."""
        problems = gsm8k_with_data.load(split="test", limit=1)
        problem = problems[0]  # Answer is 18

        assert gsm8k_with_data.evaluate_response(problem, "#### 18") is True
        assert gsm8k_with_data.evaluate_response(problem, "The answer is 18") is True
        assert gsm8k_with_data.evaluate_response(problem, "She makes $18 per day") is True
        assert gsm8k_with_data.evaluate_response(problem, "18") is True
        assert gsm8k_with_data.evaluate_response(problem, "18.0") is True

    def test_evaluate_response_incorrect(self, gsm8k_with_data):
        """Test evaluation of incorrect model responses."""
        problems = gsm8k_with_data.load(split="test", limit=1)
        problem = problems[0]  # Answer is 18

        assert gsm8k_with_data.evaluate_response(problem, "#### 20") is False
        assert gsm8k_with_data.evaluate_response(problem, "The answer is 99") is False

    def test_evaluate_response_no_number(self, gsm8k_with_data):
        """Test evaluation when model response has no extractable number."""
        problems = gsm8k_with_data.load(split="test", limit=1)
        problem = problems[0]

        assert gsm8k_with_data.evaluate_response(problem, "I don't know") is False
        assert gsm8k_with_data.evaluate_response(problem, "") is False

    def test_evaluate_large_number(self, gsm8k_with_data):
        """Test evaluation with comma-separated large number."""
        problems = gsm8k_with_data.load(split="test")
        # Problem 2: answer is 70000
        problem = problems[2]

        assert gsm8k_with_data.evaluate_response(problem, "#### 70,000") is True
        assert gsm8k_with_data.evaluate_response(problem, "The answer is 70000") is True
        assert gsm8k_with_data.evaluate_response(problem, "$70,000") is True

    def test_evaluate_decimal_answer(self, gsm8k_with_data):
        """Test evaluation with decimal answer."""
        problems = gsm8k_with_data.load(split="test")
        # Problem 4: answer is 19.5
        problem = problems[4]

        assert gsm8k_with_data.evaluate_response(problem, "#### 19.5") is True
        assert gsm8k_with_data.evaluate_response(problem, "The answer is $19.50") is True

    # ---- Subset selection ----

    def test_subset_selection(self, gsm8k_with_data):
        """Test selecting a subset of problems (e.g., first 150 for mobile eval)."""
        # Simulate selecting a subset for mobile evaluation
        problems = gsm8k_with_data.load(split="test", limit=3)

        assert len(problems) == 3
        assert problems[0].task_id == "GSM8K/0"
        assert problems[2].task_id == "GSM8K/2"

    # ---- get_sample ----

    def test_get_sample(self, gsm8k_with_data):
        """Test get_sample method."""
        sample = gsm8k_with_data.get_sample(n=2)

        assert len(sample) == 2
        assert all(isinstance(p, Problem) for p in sample)

    # ---- get_problem_by_index ----

    def test_get_problem_by_index(self, gsm8k_with_data):
        """Test getting a specific problem by index."""
        problem = gsm8k_with_data.get_problem_by_index(1)

        assert problem is not None
        assert problem.task_id == "GSM8K/1"
        assert "robe" in problem.prompt.lower()

    def test_get_problem_by_index_out_of_range(self, gsm8k_with_data):
        """Test getting a problem with out-of-range index."""
        problem = gsm8k_with_data.get_problem_by_index(9999)

        assert problem is None

    def test_get_problem_by_index_not_downloaded(self, gsm8k_dataset):
        """Test getting a problem when dataset not downloaded."""
        problem = gsm8k_dataset.get_problem_by_index(0)

        assert problem is None

    # ---- repr ----

    def test_repr(self, gsm8k_dataset):
        """Test string representation."""
        repr_str = repr(gsm8k_dataset)

        assert "GSM8KDataset" in repr_str
        assert "gsm8k" in repr_str

    # ---- validate_problem ----

    def test_validate_problem(self, gsm8k_with_data):
        """Test problem validation for GSM8K problems."""
        problems = gsm8k_with_data.load(split="test", limit=1)
        problem = problems[0]

        assert gsm8k_with_data.validate_problem(problem) is True

    # ---- Download (mocked) ----

    @patch("datasets.load_dataset")
    def test_download_success(self, mock_load_dataset, gsm8k_dataset):
        """Test successful download with mocked HuggingFace datasets."""
        # Mock the datasets library
        mock_test = [
            {"question": "What is 1+1?", "answer": "1+1=2\n#### 2"},
            {"question": "What is 2+3?", "answer": "2+3=5\n#### 5"},
        ]
        mock_train = [
            {"question": "What is 3+4?", "answer": "3+4=7\n#### 7"},
        ]

        mock_ds = {"test": mock_test, "train": mock_train}
        mock_load_dataset.return_value = mock_ds

        result = gsm8k_dataset.download()

        assert result is True
        assert (gsm8k_dataset.data_dir / "gsm8k_test.json").exists()
        assert (gsm8k_dataset.data_dir / "gsm8k_train.json").exists()
        assert (gsm8k_dataset.data_dir / "gsm8k_sample.json").exists()

        # Verify saved data
        with open(gsm8k_dataset.data_dir / "gsm8k_test.json") as f:
            saved_test = json.load(f)
        assert len(saved_test) == 2
        assert saved_test[0]["question"] == "What is 1+1?"

    @patch("datasets.load_dataset")
    def test_download_failure(self, mock_load_dataset, gsm8k_dataset):
        """Test download failure handling."""
        mock_load_dataset.side_effect = Exception("Network error")

        result = gsm8k_dataset.download()

        assert result is False


# ============================================================================
# Registry Integration Tests
# ============================================================================


class TestGSM8KRegistration:
    """Tests for GSM8K integration with the DatasetRegistry."""

    def test_gsm8k_is_registered(self):
        """Test that GSM8K dataset is automatically registered."""
        assert DatasetRegistry.is_registered("gsm8k")

    def test_gsm8k_in_list(self):
        """Test that GSM8K appears in dataset listing."""
        datasets = DatasetRegistry.list_datasets()
        assert "gsm8k" in datasets

        names = DatasetRegistry.list_names()
        assert "gsm8k" in names

    def test_create_gsm8k_via_registry(self, temp_dir):
        """Test creating GSM8K dataset through the registry."""
        dataset = DatasetRegistry.create("gsm8k", temp_dir)

        assert isinstance(dataset, GSM8KDataset)
        assert dataset.data_dir == temp_dir

    def test_gsm8k_alongside_other_datasets(self):
        """Test that GSM8K coexists with MBPP and HumanEval."""
        names = DatasetRegistry.list_names()

        assert "mbpp" in names
        assert "humaneval" in names
        assert "gsm8k" in names
        assert DatasetRegistry.count() >= 3

    def test_problem_compatibility_with_other_datasets(self, temp_dir):
        """Test that GSM8K problems have the same structure as MBPP/HumanEval."""
        # Create GSM8K data
        test_data = [
            {
                "question": "What is 2+2?",
                "answer": "2+2=4\n#### 4",
            }
        ]
        test_path = temp_dir / "gsm8k_test.json"
        with open(test_path, 'w') as f:
            json.dump(test_data, f)

        dataset = DatasetRegistry.create("gsm8k", temp_dir)
        problems = dataset.load(split="test")

        assert len(problems) == 1
        problem = problems[0]

        # Verify it has all required Problem fields
        assert hasattr(problem, "task_id")
        assert hasattr(problem, "prompt")
        assert hasattr(problem, "canonical_solution")
        assert hasattr(problem, "test_cases")
        assert hasattr(problem, "entry_point")
        assert hasattr(problem, "metadata")

        # All fields should be populated
        assert problem.task_id
        assert problem.prompt
        assert problem.canonical_solution
        assert problem.test_cases
        assert problem.entry_point


# ============================================================================
# System Prompt Test
# ============================================================================


class TestGSM8KSystemPrompt:
    """Tests for the GSM8K system prompt."""

    def test_system_prompt_exists(self):
        """Test that the system prompt is defined."""
        assert GSM8KDataset.SYSTEM_PROMPT
        assert len(GSM8KDataset.SYSTEM_PROMPT) > 0

    def test_system_prompt_mentions_python(self):
        """Test that system prompt mentions Python code execution."""
        assert "Python" in GSM8KDataset.SYSTEM_PROMPT
        assert "run_python_code" in GSM8KDataset.SYSTEM_PROMPT

    def test_system_prompt_mentions_math(self):
        """Test that system prompt mentions math problem solving."""
        assert "math" in GSM8KDataset.SYSTEM_PROMPT.lower()
