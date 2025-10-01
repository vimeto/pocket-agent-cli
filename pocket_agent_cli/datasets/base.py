"""Base classes for benchmark datasets."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any, Optional


@dataclass
class Problem:
    """Universal problem representation for code generation benchmarks.

    This dataclass provides a unified interface for problems across different
    datasets (MBPP, HumanEval, etc.).

    Attributes:
        task_id: Unique identifier for the problem (e.g., "1", "HumanEval/0")
        prompt: The problem description or function signature with docstring
        canonical_solution: The reference/expected solution
        test_cases: List of test assertions or test code
        entry_point: The function name that should be called
        metadata: Optional additional dataset-specific metadata
    """
    task_id: str
    prompt: str
    canonical_solution: str
    test_cases: List[str]
    entry_point: str
    metadata: Optional[Dict[str, Any]] = field(default_factory=dict)

    def __post_init__(self):
        """Ensure metadata is always a dict."""
        if self.metadata is None:
            self.metadata = {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert problem to dictionary representation."""
        return {
            "task_id": self.task_id,
            "prompt": self.prompt,
            "canonical_solution": self.canonical_solution,
            "test_cases": self.test_cases,
            "entry_point": self.entry_point,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Problem":
        """Create a Problem from dictionary representation."""
        return cls(
            task_id=str(data["task_id"]),
            prompt=data["prompt"],
            canonical_solution=data["canonical_solution"],
            test_cases=data["test_cases"],
            entry_point=data["entry_point"],
            metadata=data.get("metadata", {}),
        )


class Dataset(ABC):
    """Abstract base class for benchmark datasets.

    All dataset implementations must inherit from this class and implement
    the required abstract methods.

    Example:
        class MyDataset(Dataset):
            name = "my_dataset"
            description = "My custom dataset"

            def load(self, split="test", limit=None):
                # Load and return problems
                ...

            def download(self, data_dir=None):
                # Download dataset files
                ...
    """

    # Subclasses must define these class attributes
    name: str = ""
    description: str = ""
    url: str = ""

    def __init__(self, data_dir: Path):
        """Initialize dataset with data directory.

        Args:
            data_dir: Directory where dataset files are stored
        """
        self.data_dir = Path(data_dir)

    @abstractmethod
    def load(self, split: str = "test", limit: Optional[int] = None) -> List[Problem]:
        """Load problems from the dataset.

        Args:
            split: Dataset split to load ("test", "train", "sample", etc.)
            limit: Maximum number of problems to load (None for all)

        Returns:
            List of Problem objects

        Raises:
            FileNotFoundError: If dataset files don't exist
            ValueError: If split is invalid
        """
        pass

    @abstractmethod
    def download(self, data_dir: Optional[Path] = None) -> bool:
        """Download the dataset to the specified directory.

        Args:
            data_dir: Directory to download to (uses self.data_dir if None)

        Returns:
            True if download successful, False otherwise
        """
        pass

    @abstractmethod
    def is_downloaded(self, data_dir: Optional[Path] = None) -> bool:
        """Check if dataset is already downloaded.

        Args:
            data_dir: Directory to check (uses self.data_dir if None)

        Returns:
            True if dataset files exist, False otherwise
        """
        pass

    @property
    @abstractmethod
    def problem_count(self) -> int:
        """Total number of problems in the dataset.

        Returns:
            Number of problems in the full dataset
        """
        pass

    @property
    def available_splits(self) -> List[str]:
        """List of available dataset splits.

        Returns:
            List of split names (e.g., ["test", "train", "sample"])
        """
        return ["test", "sample"]

    def get_sample(self, n: int = 3) -> List[Problem]:
        """Get a small sample of problems for testing.

        Args:
            n: Number of problems to return

        Returns:
            List of n Problem objects
        """
        return self.load(split="sample", limit=n)

    def validate_problem(self, problem: Problem) -> bool:
        """Validate that a problem has all required fields.

        Args:
            problem: Problem to validate

        Returns:
            True if problem is valid, False otherwise
        """
        if not problem.task_id:
            return False
        if not problem.prompt:
            return False
        if not problem.entry_point:
            return False
        if not problem.test_cases:
            return False
        return True

    def __repr__(self) -> str:
        """String representation of dataset."""
        return f"{self.__class__.__name__}(name='{self.name}', data_dir='{self.data_dir}')"
