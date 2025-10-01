"""HumanEval dataset implementation."""

import json
import gzip
import re
from pathlib import Path
from typing import List, Optional, Dict, Any

from .base import Dataset, Problem
from .registry import DatasetRegistry


@DatasetRegistry.register("humaneval")
class HumanEvalDataset(Dataset):
    """HumanEval dataset from OpenAI.

    HumanEval is a benchmark of 164 hand-written Python programming problems
    designed to evaluate code generation capabilities of language models.

    Each problem includes:
    - A function signature with docstring (prompt)
    - A canonical solution
    - A check() function with test assertions

    Dataset source: https://github.com/openai/human-eval

    Splits:
        - test: All 164 problems
        - sample: First 5 problems (for quick testing)
    """

    name = "humaneval"
    description = "164 hand-written Python programming problems from OpenAI"
    url = "https://github.com/openai/human-eval/raw/master/data/HumanEval.jsonl.gz"

    def __init__(self, data_dir: Path):
        """Initialize HumanEval dataset.

        Args:
            data_dir: Directory where dataset files are stored
        """
        super().__init__(data_dir)

    @property
    def problem_count(self) -> int:
        """Total number of problems in HumanEval dataset."""
        return 164

    @property
    def available_splits(self) -> List[str]:
        """Available dataset splits."""
        return ["test", "sample"]

    def is_downloaded(self, data_dir: Optional[Path] = None) -> bool:
        """Check if HumanEval dataset is downloaded.

        Args:
            data_dir: Directory to check (uses self.data_dir if None)

        Returns:
            True if HumanEval data file exists
        """
        data_dir = Path(data_dir) if data_dir else self.data_dir
        return (data_dir / "humaneval.json").exists()

    def download(self, data_dir: Optional[Path] = None) -> bool:
        """Download HumanEval dataset from GitHub.

        Args:
            data_dir: Directory to download to (uses self.data_dir if None)

        Returns:
            True if download successful, False otherwise
        """
        import requests

        data_dir = Path(data_dir) if data_dir else self.data_dir
        data_dir.mkdir(parents=True, exist_ok=True)

        try:
            print(f"Downloading HumanEval dataset from {self.url}...")
            response = requests.get(self.url, timeout=60)
            response.raise_for_status()

            # Decompress gzipped content
            decompressed = gzip.decompress(response.content).decode('utf-8')

            # Parse JSONL format
            raw_problems = []
            for line in decompressed.strip().split('\n'):
                if line.strip():
                    raw_problems.append(json.loads(line))

            print(f"Downloaded {len(raw_problems)} problems")

            # Convert to our format
            converted = []
            for item in raw_problems:
                converted.append({
                    'task_id': item['task_id'],
                    'prompt': item['prompt'],
                    'canonical_solution': item['canonical_solution'],
                    'test': item['test'],
                    'entry_point': item['entry_point'],
                })

            # Save full dataset
            full_path = data_dir / "humaneval.json"
            with open(full_path, 'w') as f:
                json.dump(converted, f, indent=2)
            print(f"Saved dataset to {full_path}")

            # Save sample (first 5 problems)
            sample_path = data_dir / "humaneval_sample.json"
            with open(sample_path, 'w') as f:
                json.dump(converted[:5], f, indent=2)
            print(f"Saved sample (5 problems) to {sample_path}")

            return True

        except requests.RequestException as e:
            print(f"Download failed: {e}")
            return False
        except gzip.BadGzipFile as e:
            print(f"Failed to decompress HumanEval data: {e}")
            return False
        except json.JSONDecodeError as e:
            print(f"Failed to parse HumanEval data: {e}")
            return False
        except Exception as e:
            print(f"Unexpected error during download: {e}")
            return False

    def load(self, split: str = "test", limit: Optional[int] = None) -> List[Problem]:
        """Load HumanEval problems.

        Args:
            split: Dataset split ("test" or "sample")
            limit: Maximum number of problems to load

        Returns:
            List of Problem objects

        Raises:
            FileNotFoundError: If dataset files don't exist
            ValueError: If split is invalid
        """
        if split not in self.available_splits:
            raise ValueError(
                f"Invalid split '{split}'. Available: {self.available_splits}"
            )

        # Determine which file to load
        if split == "sample":
            data_file = self.data_dir / "humaneval_sample.json"
            if not data_file.exists():
                # Fall back to full file with limit
                data_file = self.data_dir / "humaneval.json"
                if limit is None:
                    limit = 5
        else:
            data_file = self.data_dir / "humaneval.json"

        if not data_file.exists():
            raise FileNotFoundError(
                f"HumanEval dataset not found at {data_file}. "
                f"Run download() first."
            )

        with open(data_file) as f:
            raw_data = json.load(f)

        # Apply limit
        if limit is not None:
            raw_data = raw_data[:limit]

        # Convert to Problem objects
        problems = []
        for item in raw_data:
            problem = self._convert_to_problem(item)
            problems.append(problem)

        return problems

    def _convert_to_problem(self, item: Dict[str, Any]) -> Problem:
        """Convert raw HumanEval item to Problem object.

        Args:
            item: Raw HumanEval problem dictionary

        Returns:
            Problem object
        """
        # Extract individual test cases from the check() function
        test_cases = self._extract_test_cases(item['test'], item['entry_point'])

        return Problem(
            task_id=item['task_id'],
            prompt=item['prompt'],
            canonical_solution=item['canonical_solution'],
            test_cases=test_cases,
            entry_point=item['entry_point'],
            metadata={
                'source': 'humaneval',
                'original_test': item['test'],
            }
        )

    def _extract_test_cases(self, test_code: str, entry_point: str) -> List[str]:
        """Extract individual test cases from HumanEval's check() function.

        HumanEval tests are structured as:
            def check(candidate):
                assert candidate(...) == ...
                assert candidate(...) == ...

        We extract each assertion and replace 'candidate' with the actual
        function name.

        Args:
            test_code: The complete test code with check() function
            entry_point: The function name to call

        Returns:
            List of individual test assertions
        """
        # Extract assert statements
        assertions = re.findall(r'assert\s+.+', test_code)

        # Replace 'candidate' with the actual function name
        converted = []
        for assertion in assertions:
            # Replace candidate with the entry point
            converted_assertion = assertion.replace('candidate', entry_point)
            converted.append(converted_assertion)

        # If no assertions found, return the full check function call
        if not converted:
            return [f"check({entry_point})"]

        return converted

    def create_test_harness(self, problem: Problem, generated_code: str) -> str:
        """Create complete test code for HumanEval problem.

        HumanEval requires running the check() function that contains
        all test assertions.

        Args:
            problem: The problem being tested
            generated_code: The generated solution code

        Returns:
            Complete Python code to run tests
        """
        original_test = problem.metadata.get('original_test', '')

        return f'''{generated_code}

{original_test}

check({problem.entry_point})
'''

    def get_problem_by_id(self, task_id: str) -> Optional[Problem]:
        """Get a specific problem by task ID.

        Args:
            task_id: HumanEval task ID (e.g., "HumanEval/0")

        Returns:
            Problem if found, None otherwise
        """
        try:
            problems = self.load(split="test")
            for problem in problems:
                if problem.task_id == task_id:
                    return problem
        except FileNotFoundError:
            pass
        return None
