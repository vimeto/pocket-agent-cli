"""GSM8K (Grade School Math 8K) dataset implementation."""

import json
import re
from pathlib import Path
from typing import List, Optional, Dict, Any

from .base import Dataset, Problem
from .registry import DatasetRegistry


def extract_numeric_answer(text: str) -> Optional[float]:
    """Extract a numeric answer from text in various formats.

    Handles:
        - GSM8K format: ``#### 42``
        - Natural language: ``The answer is 42``
        - Equation result: ``= 42``
        - Bare number: ``42``
        - Decimals: ``3.14``
        - Negative numbers: ``-42``
        - Comma-separated: ``1,234``
        - Dollar amounts: ``$42``
        - Trailing period: ``42.``

    Args:
        text: Text potentially containing a numeric answer.

    Returns:
        Extracted number as a float, or None if no number found.
    """
    if not text or not text.strip():
        return None

    text = text.strip()

    # 1. GSM8K format: #### <number>
    match = re.search(r'####\s*([−\-]?\s*[\d,]+\.?\d*)', text)
    if match:
        return _parse_number(match.group(1))

    # 2. "The answer is <number>" (case-insensitive)
    match = re.search(
        r'[Tt]he\s+(?:final\s+)?answer\s+is\s*:?\s*\$?\s*([−\-]?\s*[\d,]+\.?\d*)',
        text,
    )
    if match:
        return _parse_number(match.group(1))

    # 3. "= <number>" at end of text or line
    match = re.search(r'=\s*\$?\s*([−\-]?\s*[\d,]+\.?\d*)\s*$', text, re.MULTILINE)
    if match:
        return _parse_number(match.group(1))

    # 4. Last number in text (common fallback)
    matches = re.findall(r'[−\-]?\s*\$?\s*[\d,]+\.?\d*', text)
    if matches:
        # Take the last match
        candidate = matches[-1]
        parsed = _parse_number(candidate)
        if parsed is not None:
            return parsed

    return None


def _parse_number(s: str) -> Optional[float]:
    """Parse a number string, handling commas, whitespace, dollar signs, and unicode minus.

    Args:
        s: String representation of a number.

    Returns:
        Parsed float, or None if parsing fails.
    """
    if not s:
        return None
    # Clean up the string
    s = s.replace(',', '').replace('$', '').replace(' ', '')
    # Handle unicode minus sign
    s = s.replace('\u2212', '-')
    # Remove trailing period (e.g. "42.")
    if s.endswith('.'):
        s = s[:-1]
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def numeric_answers_match(answer1: float, answer2: float, tolerance: float = 1e-6) -> bool:
    """Compare two numeric answers for equality within tolerance.

    Handles integer comparison exactly (42.0 == 42) and floating point
    comparison with a small tolerance.

    Args:
        answer1: First numeric answer.
        answer2: Second numeric answer.
        tolerance: Allowed difference for floating point comparison.

    Returns:
        True if the answers match.
    """
    # Exact integer comparison
    if answer1 == int(answer1) and answer2 == int(answer2):
        return int(answer1) == int(answer2)
    # Float comparison with tolerance
    return abs(answer1 - answer2) < tolerance


@DatasetRegistry.register("gsm8k")
class GSM8KDataset(Dataset):
    """GSM8K (Grade School Math 8K) dataset.

    GSM8K is a benchmark of 8.5K grade school math word problems created by
    human problem writers. Each problem takes 2-8 steps to solve and requires
    a sequence of elementary calculations using basic arithmetic operations.

    The agent can use the ``run_python_code`` tool as a calculator to solve
    problems step by step.

    Dataset source: https://huggingface.co/datasets/openai/gsm8k

    Splits:
        - test: 1319 problems (standard evaluation set)
        - train: 7473 problems (training set)
        - sample: First 5 problems (for quick testing)
    """

    name = "gsm8k"
    description = "1319 grade school math word problems from OpenAI (test split)"
    url = "https://huggingface.co/datasets/openai/gsm8k"

    # System prompt for agent-mode math solving
    SYSTEM_PROMPT = (
        "You are a math problem solver. You have access to a Python code execution tool.\n"
        "For each problem:\n"
        "1. Think through the problem step by step\n"
        "2. Write Python code to compute the answer\n"
        "3. Submit your final numeric answer\n\n"
        "Use the run_python_code tool to verify your calculations."
    )

    TEST_SPLIT_SIZE = 1319
    TRAIN_SPLIT_SIZE = 7473

    def __init__(self, data_dir: Path):
        """Initialize GSM8K dataset.

        Args:
            data_dir: Directory where dataset files are stored.
        """
        super().__init__(data_dir)

    @property
    def problem_count(self) -> int:
        """Total number of problems in GSM8K test split."""
        return self.TEST_SPLIT_SIZE

    @property
    def available_splits(self) -> List[str]:
        """Available dataset splits."""
        return ["test", "train", "sample"]

    def is_downloaded(self, data_dir: Optional[Path] = None) -> bool:
        """Check if GSM8K dataset is downloaded.

        Args:
            data_dir: Directory to check (uses self.data_dir if None).

        Returns:
            True if GSM8K data file exists.
        """
        data_dir = Path(data_dir) if data_dir else self.data_dir
        return (data_dir / "gsm8k_test.json").exists()

    def download(self, data_dir: Optional[Path] = None) -> bool:
        """Download GSM8K dataset from HuggingFace.

        Uses the ``datasets`` library to fetch from ``openai/gsm8k``.

        Args:
            data_dir: Directory to download to (uses self.data_dir if None).

        Returns:
            True if download successful, False otherwise.
        """
        data_dir = Path(data_dir) if data_dir else self.data_dir
        data_dir.mkdir(parents=True, exist_ok=True)

        try:
            from datasets import load_dataset

            print(f"Downloading GSM8K dataset from HuggingFace ({self.url})...")
            ds = load_dataset("openai/gsm8k", "main")

            # Save test split
            test_data = [
                {"question": item["question"], "answer": item["answer"]}
                for item in ds["test"]
            ]
            test_path = data_dir / "gsm8k_test.json"
            with open(test_path, 'w') as f:
                json.dump(test_data, f, indent=2)
            print(f"Saved test split ({len(test_data)} problems) to {test_path}")

            # Save train split
            train_data = [
                {"question": item["question"], "answer": item["answer"]}
                for item in ds["train"]
            ]
            train_path = data_dir / "gsm8k_train.json"
            with open(train_path, 'w') as f:
                json.dump(train_data, f, indent=2)
            print(f"Saved train split ({len(train_data)} problems) to {train_path}")

            # Save sample (first 5 from test)
            sample_path = data_dir / "gsm8k_sample.json"
            with open(sample_path, 'w') as f:
                json.dump(test_data[:5], f, indent=2)
            print(f"Saved sample (5 problems) to {sample_path}")

            return True

        except ImportError:
            print(
                "Error: 'datasets' library is required for GSM8K download. "
                "Install with: pip install datasets"
            )
            return False
        except Exception as e:
            print(f"Download failed: {e}")
            return False

    def load(self, split: str = "test", limit: Optional[int] = None) -> List[Problem]:
        """Load GSM8K problems.

        Args:
            split: Dataset split ("test", "train", or "sample").
            limit: Maximum number of problems to load.

        Returns:
            List of Problem objects.

        Raises:
            FileNotFoundError: If dataset files don't exist.
            ValueError: If split is invalid.
        """
        if split not in self.available_splits:
            raise ValueError(
                f"Invalid split '{split}'. Available: {self.available_splits}"
            )

        raw_data = self._load_raw_data(split)

        # Apply limit
        if limit is not None:
            raw_data = raw_data[:limit]

        # Convert to Problem objects
        problems = []
        for idx, item in enumerate(raw_data):
            problem = self._convert_to_problem(item, idx)
            problems.append(problem)

        return problems

    def _load_raw_data(self, split: str) -> List[Dict[str, Any]]:
        """Load raw GSM8K data from files.

        Args:
            split: Dataset split to load.

        Returns:
            List of raw problem dictionaries.
        """
        if split == "sample":
            # Try sample file first, then fall back to test with limit
            sample_file = self.data_dir / "gsm8k_sample.json"
            if sample_file.exists():
                with open(sample_file) as f:
                    return json.load(f)
            # Fall back to test split
            test_file = self.data_dir / "gsm8k_test.json"
            if test_file.exists():
                with open(test_file) as f:
                    data = json.load(f)
                return data[:5]

        data_file = self.data_dir / f"gsm8k_{split}.json"
        if data_file.exists():
            with open(data_file) as f:
                return json.load(f)

        raise FileNotFoundError(
            f"GSM8K dataset not found in {self.data_dir}. "
            f"Run download() first or check the data directory."
        )

    def _convert_to_problem(self, item: Dict[str, Any], index: int) -> Problem:
        """Convert raw GSM8K item to Problem object.

        Args:
            item: Raw GSM8K problem dictionary with ``question`` and ``answer`` keys.
            index: Zero-based index of the problem in the split.

        Returns:
            Problem object.
        """
        # Extract the final numeric answer from the answer field
        ground_truth = self._extract_ground_truth(item["answer"])

        # Build a test assertion that checks the numeric answer
        # The test_case stores the expected numeric answer for evaluation
        test_case = f"EXPECTED_ANSWER: {ground_truth}"

        return Problem(
            task_id=f"GSM8K/{index}",
            prompt=item["question"],
            canonical_solution=item["answer"],
            test_cases=[test_case],
            entry_point="solve",  # Nominal; GSM8K doesn't have function entry points
            metadata={
                "source": "gsm8k",
                "ground_truth_answer": ground_truth,
                "full_solution": item["answer"],
            },
        )

    @staticmethod
    def _extract_ground_truth(answer_text: str) -> float:
        """Extract the ground truth numeric answer from GSM8K answer format.

        GSM8K answers end with ``#### <number>``.

        Args:
            answer_text: The full answer text from the dataset.

        Returns:
            The numeric answer.

        Raises:
            ValueError: If the answer format is unexpected.
        """
        match = re.search(r'####\s*([−\-]?\s*[\d,]+\.?\d*)', answer_text)
        if match:
            result = _parse_number(match.group(1))
            if result is not None:
                return result
        raise ValueError(f"Could not extract answer from: {answer_text!r}")

    def evaluate_response(self, problem: Problem, model_response: str) -> bool:
        """Evaluate whether a model response contains the correct answer.

        Args:
            problem: The GSM8K problem.
            model_response: The model's full response text.

        Returns:
            True if the extracted answer matches the ground truth.
        """
        ground_truth = problem.metadata.get("ground_truth_answer")
        if ground_truth is None:
            return False

        extracted = extract_numeric_answer(model_response)
        if extracted is None:
            return False

        return numeric_answers_match(extracted, ground_truth)

    def get_problem_by_index(self, index: int, split: str = "test") -> Optional[Problem]:
        """Get a specific problem by index.

        Args:
            index: Zero-based index of the problem.
            split: Dataset split to load from.

        Returns:
            Problem if found, None otherwise.
        """
        try:
            raw_data = self._load_raw_data(split)
            if 0 <= index < len(raw_data):
                return self._convert_to_problem(raw_data[index], index)
        except FileNotFoundError:
            pass
        return None
