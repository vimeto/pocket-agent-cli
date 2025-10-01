"""MBPP (Mostly Basic Python Problems) dataset implementation."""

import json
import re
from pathlib import Path
from typing import List, Optional, Dict, Any

from .base import Dataset, Problem
from .registry import DatasetRegistry


@DatasetRegistry.register("mbpp")
class MBPPDataset(Dataset):
    """MBPP (Mostly Basic Python Problems) dataset.

    MBPP is a benchmark consisting of around 974 crowd-sourced Python
    programming problems, designed to be solvable by entry-level programmers.

    Dataset source: https://github.com/google-research/google-research/tree/master/mbpp

    Splits:
        - full: All 974 problems
        - test: Problems 11-510 (500 problems, standard test set)
        - train: Problems 1-10 and 511-600 (for few-shot prompting)
        - sample: First 3 problems (for quick testing)
    """

    name = "mbpp"
    description = "974 crowd-sourced Python programming problems from Google Research"
    url = "https://raw.githubusercontent.com/google-research/google-research/master/mbpp/mbpp.jsonl"

    # Standard MBPP test split
    TEST_START_ID = 11
    TEST_END_ID = 510

    def __init__(self, data_dir: Path):
        """Initialize MBPP dataset.

        Args:
            data_dir: Directory where dataset files are stored
        """
        super().__init__(data_dir)
        self._cached_problems: Optional[List[Dict[str, Any]]] = None

    @property
    def problem_count(self) -> int:
        """Total number of problems in MBPP dataset."""
        return 974

    @property
    def available_splits(self) -> List[str]:
        """Available dataset splits."""
        return ["full", "test", "train", "sample"]

    def is_downloaded(self, data_dir: Optional[Path] = None) -> bool:
        """Check if MBPP dataset is downloaded.

        Args:
            data_dir: Directory to check (uses self.data_dir if None)

        Returns:
            True if any MBPP data file exists
        """
        data_dir = Path(data_dir) if data_dir else self.data_dir

        # Check for any of the possible data files
        possible_files = [
            data_dir / "mbpp_full.json",
            data_dir / "mbpp_test.json",
            data_dir / "mbpp_sample.json",
            data_dir / "mbpp.jsonl",
        ]

        return any(f.exists() for f in possible_files)

    def download(self, data_dir: Optional[Path] = None) -> bool:
        """Download MBPP dataset from GitHub.

        Args:
            data_dir: Directory to download to (uses self.data_dir if None)

        Returns:
            True if download successful, False otherwise
        """
        import requests

        data_dir = Path(data_dir) if data_dir else self.data_dir
        data_dir.mkdir(parents=True, exist_ok=True)

        try:
            print(f"Downloading MBPP dataset from {self.url}...")
            response = requests.get(self.url, timeout=60)
            response.raise_for_status()

            # Parse JSONL format
            raw_problems = []
            for line in response.text.strip().split('\n'):
                if line.strip():
                    raw_problems.append(json.loads(line))

            print(f"Downloaded {len(raw_problems)} problems")

            # Save full dataset
            full_path = data_dir / "mbpp_full.json"
            with open(full_path, 'w') as f:
                json.dump(raw_problems, f, indent=2)
            print(f"Saved full dataset to {full_path}")

            # Save test split (problems 11-510)
            test_problems = [
                p for p in raw_problems
                if self.TEST_START_ID <= p['task_id'] <= self.TEST_END_ID
            ]
            test_path = data_dir / "mbpp_test.json"
            with open(test_path, 'w') as f:
                json.dump(test_problems, f, indent=2)
            print(f"Saved test split ({len(test_problems)} problems) to {test_path}")

            # Save sample (first 3 problems)
            sample_problems = raw_problems[:3]
            sample_path = data_dir / "mbpp_sample.json"
            with open(sample_path, 'w') as f:
                json.dump(sample_problems, f, indent=2)
            print(f"Saved sample ({len(sample_problems)} problems) to {sample_path}")

            return True

        except requests.RequestException as e:
            print(f"Download failed: {e}")
            return False
        except json.JSONDecodeError as e:
            print(f"Failed to parse MBPP data: {e}")
            return False
        except Exception as e:
            print(f"Unexpected error during download: {e}")
            return False

    def load(self, split: str = "test", limit: Optional[int] = None) -> List[Problem]:
        """Load MBPP problems.

        Args:
            split: Dataset split ("full", "test", "train", "sample")
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
        raw_data = self._load_raw_data(split)

        # Apply limit
        if limit is not None:
            raw_data = raw_data[:limit]

        # Convert to Problem objects
        problems = []
        for item in raw_data:
            problem = self._convert_to_problem(item)
            problems.append(problem)

        return problems

    def _load_raw_data(self, split: str) -> List[Dict[str, Any]]:
        """Load raw MBPP data from files.

        Args:
            split: Dataset split to load

        Returns:
            List of raw problem dictionaries
        """
        # Try split-specific file first
        split_file = self.data_dir / f"mbpp_{split}.json"
        if split_file.exists():
            with open(split_file) as f:
                return json.load(f)

        # Fall back to full dataset and filter
        full_file = self.data_dir / "mbpp_full.json"
        if full_file.exists():
            with open(full_file) as f:
                all_problems = json.load(f)

            if split == "full":
                return all_problems
            elif split == "test":
                return [
                    p for p in all_problems
                    if self.TEST_START_ID <= p['task_id'] <= self.TEST_END_ID
                ]
            elif split == "train":
                return [
                    p for p in all_problems
                    if p['task_id'] < self.TEST_START_ID or p['task_id'] > self.TEST_END_ID
                ]
            elif split == "sample":
                return all_problems[:3]

        # Try legacy sample file location
        sample_file = self.data_dir / "mbpp_sample.json"
        if sample_file.exists() and split == "sample":
            with open(sample_file) as f:
                return json.load(f)

        raise FileNotFoundError(
            f"MBPP dataset not found in {self.data_dir}. "
            f"Run download() first or check the data directory."
        )

    def _convert_to_problem(self, item: Dict[str, Any]) -> Problem:
        """Convert raw MBPP item to Problem object.

        Args:
            item: Raw MBPP problem dictionary

        Returns:
            Problem object
        """
        # Extract function name from the reference solution
        entry_point = self._extract_function_name(item.get('code', ''))

        return Problem(
            task_id=str(item['task_id']),
            prompt=item['text'],
            canonical_solution=item.get('code', ''),
            test_cases=item.get('test_list', []),
            entry_point=entry_point,
            metadata={
                'source': 'mbpp',
                'original_task_id': item['task_id'],
            }
        )

    def _extract_function_name(self, code: str) -> str:
        """Extract the main function name from code.

        Args:
            code: Python source code

        Returns:
            Function name or "solution" as fallback
        """
        # Match the first function definition
        match = re.search(r'def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(', code)
        if match:
            return match.group(1)
        return "solution"

    def get_problem_by_id(self, task_id: int) -> Optional[Problem]:
        """Get a specific problem by task ID.

        Args:
            task_id: MBPP task ID (1-974)

        Returns:
            Problem if found, None otherwise
        """
        try:
            raw_data = self._load_raw_data("full")
            for item in raw_data:
                if item['task_id'] == task_id:
                    return self._convert_to_problem(item)
        except FileNotFoundError:
            pass
        return None

    def get_problems_by_ids(self, task_ids: List[int]) -> List[Problem]:
        """Get multiple problems by task IDs.

        Args:
            task_ids: List of MBPP task IDs

        Returns:
            List of Problem objects (in order of task_ids)
        """
        try:
            raw_data = self._load_raw_data("full")
            id_to_item = {item['task_id']: item for item in raw_data}

            problems = []
            for task_id in task_ids:
                if task_id in id_to_item:
                    problems.append(self._convert_to_problem(id_to_item[task_id]))

            return problems
        except FileNotFoundError:
            return []
